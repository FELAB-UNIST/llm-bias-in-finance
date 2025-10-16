import os
import pandas as pd
from tqdm.auto import tqdm
import concurrent.futures
from typing import Dict

from llm_clients import LLMClient, OpenAIClient, GeminiClient, TogetherClient, AnthropicClient, XAIClient
from utils import parse_json_from_text

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
                   ticker_path: str = "./data/sp500_final.csv",
                   evidence_path: str = "./data/evidence_corpus_view.csv",
                   ):
    
    # Set output path
    output_path = os.path.join(output_dir, f"{llm_client.short_model_id}_str_set_{set_number}.csv")
    
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
    
    # Process prompts in parallel
    def process_prompt(prompt):  
        return llm_client.get_response(prompt)

    results_text = [None] * len(prompts_to_run)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_prompt, prompt): idx  
            for idx, prompt in enumerate(prompts_to_run)
        }
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(prompts_to_run), desc="LLM Inference"):
            idx = futures[fut] 
            try:
                results_text[idx] = fut.result()
            except Exception as e:
                results_text[idx] = f"API_ERROR: {e}"
    
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

# ────────────── Main Execution ──────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run equal evidence experiment with different LLMs")
    parser.add_argument("--api", type=str, required=True, 
                       choices=["openai", "gemini", "together", "anthropic", "xai"],
                       help="Which API to use")
    parser.add_argument("--model-id", type=str, default=None,
                       help="Specific model ID (optional)")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for generation")
    parser.add_argument("--max-workers", type=int, default=10,
                       help="Maximum number of concurrent workers")
    parser.add_argument("--output-dir", type=str, default="./result",
                       help="Directory to save the output CSV file")
    parser.add_argument("--num-sets", type=int, default=3,
                       help="Number of experiment sets to run")
    parser.add_argument("--num-trials", type=int, default=5,
                       help="Number of trials per experiment set")
    args = parser.parse_args()
    
    # Create client based on model choice
    if args.api == "openai":
        model_id = args.model_id
        client = OpenAIClient(model_id=model_id, temperature=args.temperature)
    elif args.api == "gemini":
        model_id = args.model_id
        client = GeminiClient(model_id=model_id, temperature=args.temperature)
    elif args.api == "together":
        model_id = args.model_id
        client = TogetherClient(model_id=model_id, temperature=args.temperature)
    elif args.api == "anthropic":
        model_id = args.model_id
        client = AnthropicClient(model_id=model_id, temperature=args.temperature)
    elif args.api == "xai":
        model_id = args.model_id
        client = XAIClient(model_id=model_id, temperature=args.temperature)
    
    for i in range(1, args.num_sets + 1):
        print(f"\n{'─'*20} Running Set {i}/{args.num_sets} {'─'*20}")
        run_experiment(
            llm_client=client, 
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            set_number=i,
        )