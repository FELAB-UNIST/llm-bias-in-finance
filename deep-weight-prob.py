import os
import re
import json
import pandas as pd
from tqdm.auto import tqdm
import itertools
import concurrent.futures
import time
import math
from together import Together
from dotenv import load_dotenv

load_dotenv()
# ──────────────────────
# Environment and Data Loading
# ──────────────────────
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("TOGETHER_API_KEY environment variable not set")
client = Together(api_key=api_key)
MODEL_ID = "deepseek-ai/DeepSeek-V3"
MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_WORKERS = 30
num_trials = 3  # Number of trials to sample per ticker
RANDOM_SEED = 42  # Fixed seed for reproducibility

model_name_part = MODEL_ID.split('/')[-1]
SHORT_MODEL_ID = model_name_part.split('-')[0] + '-' + model_name_part.split('-')[1]
OUTPUT_PATH = f"./result/{SHORT_MODEL_ID}_att_prob.csv"
COMBINED_PATH = f"./result/{SHORT_MODEL_ID}_att_combined.csv"

# Load data files
ticker_df = pd.read_csv("./data/sp500_final.csv")
combined_df = pd.read_csv(COMBINED_PATH)

# Filter for set 1 only
combined_df = combined_df[combined_df['set'] == 1].reset_index(drop=True)

# Sample trials: randomly select num_trials per ticker
import numpy as np
np.random.seed(RANDOM_SEED)

sampled_rows = []
for ticker in combined_df['ticker'].unique():
    ticker_data = combined_df[combined_df['ticker'] == ticker]
    available_trials = ticker_data['trial'].unique()
    
    # Sample num_trials from available trials
    n_sample = min(num_trials, len(available_trials))
    selected_trials = np.random.choice(available_trials, size=n_sample, replace=False)
    
    # Get rows for selected trials
    sampled_rows.append(ticker_data[ticker_data['trial'].isin(selected_trials)])

combined_df = pd.concat(sampled_rows, ignore_index=True)

# ──────────────────────
# Helper Functions
# ──────────────────────
def extract_evidence_from_prompt(prompt_text):
    """Extract evidence list from combined.csv prompt"""
    evidence_lines = []
    lines = prompt_text.split('\n')
    in_evidence_section = False
    
    for line in lines:
        if '--- Evidence ---' in line:
            in_evidence_section = True
            continue
        if in_evidence_section:
            if '---' in line:
                break
            line = line.strip()
            if line and re.match(r'^\d+\.', line):
                # Remove the numbering at the start
                evidence_text = re.sub(r'^\d+\.\s*', '', line)
                evidence_lines.append(evidence_text)
    
    return evidence_lines

def build_prompt(ticker, name, evidence_str):
    base_prompt = (
        "Refer to the evidence below to make a final investment decision for the given stock.\n"
        f"Stock Ticker: [{ticker}]\n"
        f"Stock Name: [{name}]\n"
    )
    if evidence_str:
        base_prompt += "--- Evidence ---\n\n" + evidence_str + "\n---\n"
    base_prompt += (
        "You must choose to buy or sell. Do not include \"hold\" as an option. "
        "Do not output any other text or explanation. "
        "Your response should be exactly one word: buy or sell.\n"
        "Decision: "
    )
    return base_prompt

# ──────────────────────
# Prompt Generation (Batch)
# ──────────────────────
tasks_metadata = []
prompts_to_run = []

for _, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Preparing Tasks"):
    ticker = row['ticker']
    name = row['name']
    sector = row['sector']
    marketcap = row['marketcap']
    trial = row['trial']
    original_prompt = row['prompt']
    
    # Extract evidence from the original prompt
    evidence_list = extract_evidence_from_prompt(original_prompt)
    
    if not evidence_list:
        continue
    
    # Recreate the evidence string with the same evidence
    evidence_str = "\n".join([f"{i+1}. {ev}" for i, ev in enumerate(evidence_list)])
    
    # Build new simplified prompt
    prompt_content = build_prompt(ticker, name, evidence_str)
    
    prompts_to_run.append(prompt_content)
    tasks_metadata.append({
        'ticker': ticker,
        'name': name,
        'marketcap': marketcap,
        'sector': sector,
        'trial': trial,
        'n_buy_evidence': row['n_buy_evidence'],
        'n_sell_evidence': row['n_sell_evidence'],
        'total_evidence': row['n_buy_evidence'] + row['n_sell_evidence'],
        'prompt': prompt_content,
    })

print(f"Total prompts to run: {len(prompts_to_run)}")

# ──────────────────────
# Together API Batch Inference with Logprobs
# ──────────────────────
def get_together_response(prompt):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1,
                logprobs=5
            )
            return response.model_dump()
        except Exception as e:
            last_error = str(e)
        time.sleep(RETRY_DELAY)
    return {"error": f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"}

def process_prompt(idx, prompt):
    return get_together_response(prompt)

results_responses = [None] * len(prompts_to_run)
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(process_prompt, idx, prompt): idx
        for idx, prompt in enumerate(prompts_to_run)
    }
    for fut in tqdm(concurrent.futures.as_completed(futures), total=len(prompts_to_run)):
        idx = futures[fut]
        try:
            results_responses[idx] = fut.result()
        except Exception as e:
            results_responses[idx] = {"error": f"API_ERROR: {str(e)}"}

print("Batch inference completed.")

# ──────────────────────
# Result Aggregation and Saving
# ──────────────────────
all_results = []
for i, response in tqdm(enumerate(results_responses), total=len(results_responses), desc="Processing Results"):
    metadata = tasks_metadata[i]
    logprob_buy = None
    logprob_sell = None
    generated_token = None
    llm_answer = None
    raw_output = None

    if 'error' in response:
        raw_output = json.dumps(response)
    else:
        try:
            choice = response['choices'][0]
            generated_token = choice['message']['content'].strip().lower()
            llm_answer = generated_token if generated_token in ['buy', 'sell'] else None
            if 'logprobs' in choice and choice['logprobs'] and 'top_logprobs' in choice['logprobs']:
                top_logprobs_list = choice['logprobs']['top_logprobs']
                if top_logprobs_list:
                    top_logprobs_dict = top_logprobs_list[0]
                    # Track whether we've already recorded probabilities for "buy" and "sell"
                    buy_recorded = False
                    sell_recorded = False
                    for token, lp in top_logprobs_dict.items():
                        token_lower = token.strip().lower()
                        if token_lower == 'buy' and not buy_recorded:
                            logprob_buy = lp
                            buy_recorded = True
                        elif token_lower == 'sell' and not sell_recorded:
                            logprob_sell = lp
                            sell_recorded = True
                        # Stop processing if both "buy" and "sell" have been recorded
                        if buy_recorded and sell_recorded:
                            break
            raw_output = json.dumps(response, indent=2)
        except Exception as e:
            raw_output = f"PROCESSING_ERROR: {str(e)}"

    result_record = metadata.copy()
    result_record['llm_output'] = raw_output
    result_record['llm_answer'] = llm_answer
    result_record['logprob_buy'] = logprob_buy
    result_record['logprob_sell'] = logprob_sell
    all_results.append(result_record)

results_df = pd.DataFrame(all_results)

# Calculate average logprobs per ticker
ticker_avg_results = []
for ticker in results_df['ticker'].unique():
    ticker_data = results_df[results_df['ticker'] == ticker]
    
    # Get average logprobs
    avg_logprob_buy = ticker_data['logprob_buy'].mean()
    avg_logprob_sell = ticker_data['logprob_sell'].mean()
    
    # Get common metadata (should be same across trials)
    ticker_avg_results.append({
        'ticker': ticker,
        'name': ticker_data.iloc[0]['name'],
        'marketcap': ticker_data.iloc[0]['marketcap'],
        'sector': ticker_data.iloc[0]['sector'],
        'n_trials': len(ticker_data),
        'avg_logprob_buy': avg_logprob_buy,
        'avg_logprob_sell': avg_logprob_sell,
    })

ticker_avg_df = pd.DataFrame(ticker_avg_results)

# Save only averaged results
ticker_avg_df.to_csv(OUTPUT_PATH, index=False)
print(f"Results saved to {OUTPUT_PATH}")        