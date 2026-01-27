import os
import re
import json
import time
import pandas as pd
from tqdm.auto import tqdm
import itertools
import concurrent.futures
from typing import Optional, Dict, List, Tuple
import argparse

# Import the unified LLMClient
from llm_clients import LLMClient

# ────────────── Configuration ──────────────
MAX_WORKERS = 10
num_trials = 10

# ────────────── Helper Functions ──────────────
def parse_json_from_text(text: str) -> Optional[Dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def get_evidence_list_v3(row) -> Tuple[List[str], List[str]]:
    qual_evidence = [str(row.get(f"evidence{i}_qual", '')).strip() 
                    for i in range(1, 3) 
                    if pd.notna(row.get(f"evidence{i}_qual"))]
    quat_evidence = [str(row.get(f"evidence{i}_quat", '')).strip() 
                    for i in range(1, 3) 
                    if pd.notna(row.get(f"evidence{i}_quat"))]
    return qual_evidence, quat_evidence

def build_prompt(ticker: str, name: str, evidence_str: str, decision_order: str) -> str:
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
def extract_evidence_texts(prompt: str) -> List[str]:
    """Extracts evidence texts from the prompt."""
    match = re.search(r"--- Evidence ---\n(.*?)\n---", prompt, re.DOTALL)
    if not match:
        return []
    evidence_block = match.group(1)
    parts = re.split(r"\n\d+\.\s+", "\n" + evidence_block)
    evidences = [p.strip() for p in parts if p.strip()]
    return evidences

def reconstruct_prompt(original_prompt: str, new_evidence_list: List[str]) -> str:
    """Replaces the evidence section in the prompt with new evidence list."""
    match = re.search(r"(--- Evidence ---\n)(.*?)(\n---)", original_prompt, re.DOTALL)
    if not match:
        return original_prompt
    
    start_tag = match.group(1)
    end_tag = match.group(3)
    
    import random
    random.shuffle(new_evidence_list)
    
    new_evidence_str = "\n".join([f"{i+1}. {ev}" for i, ev in enumerate(new_evidence_list)])
    
    new_prompt = original_prompt.replace(match.group(0), f"{start_tag}{new_evidence_str}{end_tag}")
    return new_prompt

def run_experiment(llm_client: LLMClient,
                  class_type: str,
                  target_group: str,
                  bias_direction: Optional[str] = None,
                  qual_evidence_path: str = "./data/evidence_corpus_qual.csv",
                  quant_evidence_path: str = "./data/evidence_corpus_quant.csv",
                  output_dir: str = "./result",
                  input_dir: Optional[str] = None):
    
    if input_dir is None:
        input_dir = output_dir

    # Set paths
    model_suffix = llm_client.short_model_id
    if llm_client.reasoning_effort:
        model_suffix = f"{model_suffix}_{llm_client.reasoning_effort}"
        
    output_path = os.path.join(output_dir, f"{model_suffix}_weight_evidence_{class_type}_{target_group}.csv")
    
    combined_path = os.path.join(input_dir, f"{model_suffix}_att_combined.csv")
    
    if not os.path.exists(combined_path):
         # Try without reasoning effort suffix if applicable
         if llm_client.reasoning_effort:
             combined_path_simple = os.path.join(input_dir, f"{llm_client.short_model_id}_att_combined.csv")
             if os.path.exists(combined_path_simple):
                 combined_path = combined_path_simple
    
    print(f"Loading combined results from: {combined_path}")
    if not os.path.exists(combined_path):
        print(f"Combined results file not found at {combined_path}")
        return

    try:
        df_combined = pd.read_csv(combined_path)
    except Exception as e:
        print(f"Error reading combined file: {e}")
        return

    # Add cap_class based on marketcap quartiles before filtering
    if 'marketcap' in df_combined.columns:
        df_combined['cap_class'] = pd.qcut(
            df_combined['marketcap'], 4, labels=['Q4', 'Q3', 'Q2', 'Q1'], duplicates='drop'
        )

    # Filter by target group
    print(f"Filtering for {class_type} == {target_group}...")
    original_len = len(df_combined)
    
    if class_type == 'sector':
        if 'sector' in df_combined.columns:
            df_combined = df_combined[df_combined['sector'] == target_group]
        else:
            print(f"Warning: 'sector' column not found. Available columns: {df_combined.columns.tolist()}")
            return
    elif class_type == 'size':
        # Calculate size group if not present
        if 'marketcap_group' not in df_combined.columns:
             if 'marketcap' in df_combined.columns:
                 df_combined['marketcap_group'] = pd.qcut(
                    df_combined['marketcap'], 4, labels=['Q4', 'Q3', 'Q2', 'Q1'], duplicates='drop'
                )
             else:
                 print("Warning: 'marketcap' column not found for size grouping.")
                 return
        
        df_combined = df_combined[df_combined['marketcap_group'] == target_group]
    
    elif class_type == 'bias_score':
        # Calculate bias scores per ticker
        # Map answers to 1 (buy), -1 (sell), 0 (others)
        def map_bias(x):
            s = str(x).lower()
            if 'buy' in s: return 1
            if 'sell' in s: return -1
            return 0
        
        # Group by ticker to calculate mean bias score
        ticker_scores = df_combined.groupby('ticker')['llm_answer'].apply(lambda x: x.apply(map_bias).mean())
        ticker_abs_scores = ticker_scores.abs().sort_values(ascending=False)
        
        # Determine direction per ticker
        ticker_directions = ticker_scores.apply(lambda x: 'buy' if x > 0 else ('sell' if x < 0 else 'neutral'))
        
        n_tickers = len(ticker_abs_scores)
        top_10_count = int(n_tickers * 0.1)
        
        if target_group == 'top_10_abs':
            selected_tickers = ticker_abs_scores.head(top_10_count).index
            print(f"Selected Top 10% Absolute Bias Tickers (Count: {len(selected_tickers)})")
            print(f"Score Range: {ticker_abs_scores.iloc[0]:.4f} to {ticker_abs_scores.iloc[top_10_count-1]:.4f}")
        elif target_group == 'bottom_10_abs':
            selected_tickers = ticker_abs_scores.tail(top_10_count).index
            print(f"Selected Bottom 10% Absolute Bias Tickers (Count: {len(selected_tickers)})")
            print(f"Score Range: {ticker_abs_scores.iloc[-top_10_count]:.4f} to {ticker_abs_scores.iloc[-1]:.4f}")
        else:
            print(f"Unknown target group for bias_score: {target_group}")
            return
            
        df_combined = df_combined[df_combined['ticker'].isin(selected_tickers)]
        
        # Filter rows based on ticker direction
        # We only keep rows where the llm_answer matches the ticker's determined bias direction
        print("Filtering rows to match ticker-level bias direction...")
        rows_to_keep = []
        for idx, row in df_combined.iterrows():
            ticker = row['ticker']
            direction = ticker_directions.get(ticker, 'neutral')
            current_answer = str(row.get('llm_answer', '')).lower()
            
            # Check if current answer matches the ticker's bias direction
            if direction != 'neutral' and direction in current_answer:
                rows_to_keep.append(idx)
        
        before_len = len(df_combined)
        df_combined = df_combined.loc[rows_to_keep]
        print(f"Filtered {before_len} -> {len(df_combined)} rows matching ticker bias direction.")
            
    print(f"Filtered {original_len} -> {len(df_combined)} rows based on group.")

    # Filter by bias direction (llm_answer)
    # Note: For bias_score mode, we already filtered by ticker-specific direction above.
    if bias_direction and bias_direction != 'auto' and class_type != 'bias_score':
        print(f"Filtering for llm_answer == {bias_direction}")
        before_len = len(df_combined)
        df_combined = df_combined[df_combined['llm_answer'].str.lower() == bias_direction.lower()]
        print(f"Filtered {before_len} -> {len(df_combined)} rows based on bias direction.")
    
    if df_combined.empty:
        print("No rows found for this group/direction. Skipping.")
        return

    # Load evidence corpus
    qual_evidence_df = pd.read_csv(qual_evidence_path)
    quant_evidence_df = pd.read_csv(quant_evidence_path)

    # Merge evidence dataframes to create a lookup pool
    evidence_df = pd.merge(
        qual_evidence_df,
        quant_evidence_df,
        on=['ticker', 'opinion'],
        suffixes=('_qual', '_quat')
    )
    
    # Build evidence lookup: ticker -> opinion -> list of all evidence texts
    evidence_lookup = {}
    for _, row in evidence_df.iterrows():
        ticker = row['ticker']
        opinion = row['opinion'].lower()
        if ticker not in evidence_lookup:
            evidence_lookup[ticker] = {'buy': [], 'sell': []}
        
        qual, quat = get_evidence_list_v3(row)
        evidence_lookup[ticker][opinion] = qual + quat

    # Prepare tasks
    tasks_metadata = []
    prompts_to_run = []
    
    print(f"Preparing tasks with bias_direction: {bias_direction}")

    for idx, row in tqdm(df_combined.iterrows(), total=len(df_combined), desc="Preparing Tasks"):
        ticker = row['ticker']
        original_prompt = row['prompt']
        current_answer = str(row.get('llm_answer', '')).lower()
        
        # Determine counter opinion
        if class_type == 'bias_score':
            # Use ticker-level direction
            ticker_dir = ticker_directions.get(ticker, 'neutral')
            if ticker_dir == 'buy':
                counter_opinion = 'sell'
            elif ticker_dir == 'sell':
                counter_opinion = 'buy'
            else:
                continue
        elif bias_direction and bias_direction != 'auto':
            target_opinion = bias_direction.lower()
            counter_opinion = 'sell' if target_opinion == 'buy' else 'buy'
        else:
            # Auto mode: determine based on current answer
            if 'buy' in current_answer:
                counter_opinion = 'sell'
            elif 'sell' in current_answer:
                counter_opinion = 'buy'
            else:
                # If answer is neither buy nor sell (e.g. hold or error), skip or default?
                # Let's skip for now as we can't counter it easily
                continue
        
        # Extract existing evidence from the prompt
        existing_evidences = extract_evidence_texts(original_prompt)
        existing_set = set(existing_evidences)
        
        # Get all available counter evidence for this ticker
        if ticker not in evidence_lookup:
            # print(f"Warning: No evidence found for ticker {ticker}")
            continue
            
        available_counter_evidences = evidence_lookup[ticker].get(counter_opinion, [])
        
        # Find unused counter evidences
        unused_counter_evidences = [ev for ev in available_counter_evidences if ev not in existing_set]
        
        # We want to create two new tasks:
        # 1. Add 1 counter evidence (Total +1)
        # 2. Add 2 counter evidences (Total +2)
        
        # Task 1: Add 1 counter evidence
        if len(unused_counter_evidences) >= 1:
            # Pick 1
            new_ev = unused_counter_evidences[0]
            new_evidence_list = existing_evidences + [new_ev]
            
            new_prompt = reconstruct_prompt(original_prompt, new_evidence_list)
            prompts_to_run.append(new_prompt)
            
            meta = row.to_dict()
            meta['prompt'] = new_prompt
            meta['added_evidence_count'] = 1
            meta['added_evidence_type'] = counter_opinion
            meta['original_prompt_idx'] = idx
            meta['original_llm_answer'] = row.get('llm_answer')
            tasks_metadata.append(meta)
            
        # Task 2: Add 2 counter evidences
        if len(unused_counter_evidences) >= 2:
            # Pick 2
            new_evs = unused_counter_evidences[:2]
            new_evidence_list = existing_evidences + new_evs
            
            new_prompt = reconstruct_prompt(original_prompt, new_evidence_list)
            prompts_to_run.append(new_prompt)
            
            meta = row.to_dict()
            meta['prompt'] = new_prompt
            meta['added_evidence_count'] = 2
            meta['added_evidence_type'] = counter_opinion
            meta['original_prompt_idx'] = idx
            meta['original_llm_answer'] = row.get('llm_answer')
            tasks_metadata.append(meta)
            
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
        
        # Get cost and tokens if available
        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        ttft = 0.0
        if hasattr(llm_client, 'last_call_cost'):
            cost = llm_client.last_call_cost
        if hasattr(llm_client, 'last_input_tokens'):
            input_tokens = llm_client.last_input_tokens
        if hasattr(llm_client, 'last_output_tokens'):
            output_tokens = llm_client.last_output_tokens
        if hasattr(llm_client, 'last_ttft'):
            ttft = llm_client.last_ttft
        
        return response, latency, cost, input_tokens, output_tokens, ttft
    
    results_text = [None] * len(prompts_to_run)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
        "model": model_suffix,
        "class_type": class_type,
        "target_group": target_group,
        "total_prompts": len(prompts_to_run),
        "total_cost_usd": round(total_cost, 4),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_time_seconds": round(total_time, 2),
        "average_latency_seconds": round(sum(individual_latencies) / len(individual_latencies), 2) if individual_latencies else 0,
        "average_ttft_seconds": round(sum(individual_ttfts) / len(individual_ttfts), 2) if individual_ttfts else 0,
    }

    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"{model_suffix}_verification_metrics_{class_type}_{target_group}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

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
    print(f"✅ Results saved to {output_path}")

# ────────────── Main Execution ──────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run weight conflict analysis with different LLMs")
    parser.add_argument("--model-id", type=str, required=True,
                       help="OpenRouter model ID (e.g., 'openai/gpt-4.1', 'anthropic/claude-sonnet-4')")
    parser.add_argument("--json-path", type=str, default=None,
                       help="Path to the JSON file containing bias results (Optional for bias_score mode)")
    parser.add_argument("--temperature", type=float, default=0.6,
                       help="Temperature for generation")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                       help="Reasoning effort level for reasoning models (e.g., o1, o3, gpt-5)")
    parser.add_argument("--max-tokens", type=int, default=None,
                       help="Maximum tokens for response (None for model default)")
    parser.add_argument("--max-workers", type=int, default=10,
                       help="Maximum number of concurrent workers")
    parser.add_argument("--num-trials", type=int, default=10,
                       help="Number of trials to run")
    parser.add_argument("--qual-evidence", type=str, default="./data/evidence_corpus_qual_mixed.csv",
                       help="Path to qualitative evidence CSV")
    parser.add_argument("--quant-evidence", type=str, default="./data/evidence_corpus_quant_mixed.csv",
                       help="Path to quantitative evidence CSV")
    parser.add_argument("--output-dir", type=str, default="./result",
                       help="Output directory for results")
    parser.add_argument("--input-dir", type=str, default=None,
                       help="Input directory for bias files (defaults to output-dir if not specified)")
    
    args = parser.parse_args()
    
    # Update global variables
    MAX_WORKERS = args.max_workers
    num_trials = args.num_trials


    # Create unified LLMClient
    client = LLMClient(
        model_id=args.model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort
    )
    
    print(f"Using model: {args.model_id}")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")
        
    # Define tasks for bias score grouping
    tasks = [
        ('bias_score', 'top_10_abs', 'auto'),
        ('bias_score', 'bottom_10_abs', 'auto')
    ]
        
    # Run experiment for each task
    for class_type, target_group, bias_direction in tasks:
        print(f"\n{'='*20} Processing {class_type}: {target_group} (Direction: {bias_direction}) {'='*20}")
        run_experiment(client,
                      class_type=class_type,
                      target_group=target_group,
                      bias_direction=bias_direction,
                      qual_evidence_path=args.qual_evidence,
                      quant_evidence_path=args.quant_evidence,
                      output_dir=args.output_dir,
                      input_dir=args.input_dir)
