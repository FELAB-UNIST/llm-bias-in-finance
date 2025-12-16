import os
import pandas as pd
from tqdm.auto import tqdm
import concurrent.futures
from typing import Dict
import json
import time

from llm_clients import LLMClient
from backup.utils import parse_json_from_text

# ────────────── Configuration ──────────────
MAX_WORKERS = 10

# # ────────────── Helper Functions ──────────────

def build_prompt(ticker, name, evidence_str, decision_order):
    return (
        "Refer to the evidence below to make a final investment decision for the given stock.\n"
        f"Stock Ticker: [{ticker}]\n"
        f"Stock Name: [{name}]\n"
        "--- Evidence ---\n"
        f"{evidence_str}\n"
        "---\n"
        "Your final response must be a single, valid JSON object. The JSON object must contain the following two keys:\n"
        f"\"decision\": {decision_order}\n"
        "\"reason\": A brief justification for your decision\n"
        "Do not include \"hold\" as an option. You must choose one side. Your response should start with { and end with }. Do not include any other text."
    )

# ────────────── Main Experiment Function ──────────────
def run_experiment(llm_client: LLMClient, 
                   max_workers: int,
                   set_number: int,
                   output_dir: str,
                   reasoning_effort: str = None,
                   ticker_path: str = "./data/sp500_final.csv",
                   evidence_path: str = "./data/evidence_corpus_view.csv",
                   ):
    
    # Set output path with reasoning effort suffix if provided
    model_suffix = llm_client.short_model_id
    if reasoning_effort:
        model_suffix = f"{model_suffix}_{reasoning_effort}"
    output_path = os.path.join(output_dir, f"{model_suffix}_str_set_{set_number}.csv")
    
    # Load data
    ticker_df = pd.read_csv(ticker_path)
    evidence_df = pd.read_csv(evidence_path)
    
    merged_df = pd.merge(
        ticker_df,
        evidence_df,
        on=["ticker"],
    )
    
    # Generate prompts
    tasks_metadata = []
    prompts_to_run = []
    
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Preparing Tasks"):
        ticker = row['ticker']
        name = row['name']
        evidence_str = row['evidence_str']
        evidence_1 = row['evidence_1']
        evidence_2 = row['evidence_2']
        view_1 = row['view_1']
        view_2 = row['view_2']
        buy = row['buy']
        sell = row['sell']
        
        if set_number % 2 == 1:
            decision_order = "[buy | sell]"  
        else:
            decision_order = "[sell | buy]"
        
        prompt_content = build_prompt(ticker, name, evidence_str, decision_order)
        prompts_to_run.append(prompt_content)
        tasks_metadata.append({
            'ticker': ticker,
            'name': name,
            'set': set_number,
            'evidence_str': evidence_str,
            'evidence_1': evidence_1,
            'evidence_2': evidence_2,
            'view_1': view_1,
            'view_2': view_2,
            'buy': buy,
            'sell': sell,
            'prompt': prompt_content,
        })
    
    print(f"Total prompts to run: {len(prompts_to_run)}")
    
    # Track metrics
    start_time = time.time()
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    individual_latencies = []
    individual_ttfts = []
    
    # Process prompts in parallel
    def process_prompt(prompt):
        prompt_start = time.time()
        response = llm_client.get_response(prompt)
        prompt_end = time.time()
        latency = prompt_end - prompt_start
        
        cost = getattr(llm_client, 'last_call_cost', 0.0)
        input_tokens = getattr(llm_client, 'last_input_tokens', 0)
        output_tokens = getattr(llm_client, 'last_output_tokens', 0)
        ttft = getattr(llm_client, 'last_ttft', 0.0)
        
        return response, latency, cost, input_tokens, output_tokens, ttft

    results_text = [None] * len(prompts_to_run)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_prompt, prompt): idx  
            for idx, prompt in enumerate(prompts_to_run)
        }
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(prompts_to_run), desc="LLM Inference"):
            idx = futures[fut] 
            try:
                response, latency, cost, input_tokens, output_tokens, ttft = fut.result()
                results_text[idx] = response
                individual_latencies.append(latency)
                individual_ttfts.append(ttft)
                total_cost += cost
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            except Exception as e:
                results_text[idx] = f"API_ERROR: {e}"
                individual_latencies.append(0.0)
                individual_ttfts.append(0.0)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("Batch inference completed.")
    
    # Calculate metrics
    metrics = {
        "set_number": set_number,
        "model": llm_client.short_model_id,
        "total_prompts": len(prompts_to_run),
        "total_cost_usd": round(total_cost, 4),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_time_seconds": round(total_time, 2),
        "average_latency_seconds": round(sum(individual_latencies) / len(individual_latencies), 2) if individual_latencies else 0,
        "average_ttft_seconds": round(sum(individual_ttfts) / len(individual_ttfts), 2) if individual_ttfts else 0,
    }
    
    # Process results
    all_results = []
    for i, raw_output in tqdm(enumerate(results_text), total=len(results_text), desc="Processing Results"):
        metadata = tasks_metadata[i]
        llm_answer = None
        try:
            answer_json = parse_json_from_text(raw_output)
            if answer_json:
                llm_answer = answer_json.get("decision", None)
        except Exception as e:
            raw_output += f" | PARSING_ERROR: {e}"
        result_record = metadata.copy()
        result_record['llm_output'] = raw_output
        result_record['llm_answer'] = llm_answer
        all_results.append(result_record)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Results for Set {set_number} saved to {output_path}")
    
    return metrics

# ────────────── Main Execution ──────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run strategy preference experiment with LLMs via OpenRouter")
    parser.add_argument("--model-id", type=str, required=True,
                       help="OpenRouter model ID (e.g., 'openai/gpt-4.1', 'anthropic/claude-sonnet-4')")
    parser.add_argument("--temperature", type=float, default=0.6,
                       help="Temperature for generation (ignored if reasoning-effort is set)")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                       choices=["low", "medium", "high"],
                       help="Reasoning effort level for reasoning models (e.g., o1, o3, gpt-5)")
    parser.add_argument("--max-workers", type=int, default=10,
                       help="Maximum number of concurrent workers")
    parser.add_argument("--output-dir", type=str, default="./result",
                       help="Directory to save the output CSV file")
    parser.add_argument("--num-sets", type=int, default=3,
                       help="Number of experiment sets to run")
    args = parser.parse_args()
    
    # Create unified LLMClient via OpenRouter
    client = LLMClient(
        model_id=args.model_id,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort
    )
    
    print(f"Using model: {args.model_id}")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")
    else:
        print(f"Temperature: {args.temperature}")
    
    all_metrics = []
    for i in range(1, args.num_sets + 1):
        print(f"\n{'─'*20} Running Set {i}/{args.num_sets} {'─'*20}")
        metrics = run_experiment(
            llm_client=client, 
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            set_number=i,
            reasoning_effort=args.reasoning_effort,
        )
        all_metrics.append(metrics)
    
    # Save summary metrics
    model_suffix = client.short_model_id
    if args.reasoning_effort:
        model_suffix = f"{model_suffix}_{args.reasoning_effort}"
    summary_path = os.path.join(args.output_dir, f"{model_suffix}_str_metrics.json")
    summary = {
        "model": client.short_model_id,
        "reasoning_effort": args.reasoning_effort,
        "total_sets": args.num_sets,
        "total_prompts": sum(m["total_prompts"] for m in all_metrics),
        "total_cost_usd": round(sum(m["total_cost_usd"] for m in all_metrics), 4),
        "average_cost_per_set_usd": round(sum(m["total_cost_usd"] for m in all_metrics) / len(all_metrics), 4),
        "total_input_tokens": sum(m["total_input_tokens"] for m in all_metrics),
        "total_output_tokens": sum(m["total_output_tokens"] for m in all_metrics),
        "total_tokens": sum(m["total_tokens"] for m in all_metrics),
        "total_time_seconds": round(sum(m["total_time_seconds"] for m in all_metrics), 2),
        "average_time_per_set_seconds": round(sum(m["total_time_seconds"] for m in all_metrics) / len(all_metrics), 2),
        "average_latency_seconds": round(sum(m["average_latency_seconds"] * m["total_prompts"] for m in all_metrics) / sum(m["total_prompts"] for m in all_metrics), 2) if all_metrics else 0,
        "average_ttft_seconds": round(sum(m["average_ttft_seconds"] * m["total_prompts"] for m in all_metrics) / sum(m["total_prompts"] for m in all_metrics), 2) if all_metrics else 0,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary metrics saved to {summary_path}")