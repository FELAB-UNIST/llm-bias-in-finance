import os
import pandas as pd
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
import sys
import os
import argparse
from llm_clients import LLMClient

# Load data
df = pd.read_csv('./data/sp500_final.csv')
OUTPUT_PATH = "./data/evidence_corpus_view.csv"

# Define view pairs
view_pairs = [
    ("momentum", "contrarian"),
]

# Create records for each ticker and view pair
records = []
for _, row in df.iterrows():
    ticker = row['ticker']
    name = row['name']
    for view_1, view_2 in view_pairs:
            # 1. Buy, Sell order
            records.append({
                "ticker": ticker,
                "name": name,
                "buy_view": view_1,  # View that gives Buy recommendation
                "view_1": view_1,
                "rec_1": "Buy",
                "view_2": view_2,
                "rec_2": "Sell",
                "buy": view_1,
                "sell": view_2,
            })
            # 2. Sell, Buy order (alternating)
            records.append({
                "ticker": ticker,
                "name": name,
                "buy_view": view_2,  # View that gives Buy recommendation
                "view_1": view_1,
                "rec_1": "Sell",
                "view_2": view_2,
                "rec_2": "Buy",
                "buy": view_2,
                "sell": view_1,
            })
    
expanded = pd.DataFrame.from_records(records)

# Prompt template for balanced conflicting evidence
BALANCED_CONFLICT_PROMPT_TEMPLATE = """
Stock Ticker: [{ticker}]
Stock Name: [{name}]

Your task is to create a pair of conflicting, but balanced, investment evidences for [{name}].

- The first evidence must support a [{rec_1}] recommendation from a [{view_1}] perspective.
- The second evidence must support a [{rec_2}] recommendation from a [{view_2}] perspective.

**Crucial Constraint: Both evidences must be based on the exact same magnitude of expected price change: 5%.**

Format your response strictly as two numbered points. **Do not include any headers, titles, or prefixes before each point.**

1. ...
2. ...

Instructions:
- Write each evidence concisely, limited to 1-2 sentences.
- Only return the 2 structured evidences as specified. Do not include any extra explanation.
- Create purely fictional but plausible evidence. Do not use real-world public information.
- Each point must explicitly state the expected price change of 5% and its reasoning.
"""



def run_batch(prompt_template, outfile_name, llm_client, max_workers, data_df):
    """Run batch processing for evidence generation"""
    
    def get_evidence(row):
        """Generate evidence for a single row"""
        prompt = prompt_template.format(
            ticker=row['ticker'],
            name=row['name'],
            rec_1=row['rec_1'],
            view_1=row['view_1'],
            rec_2=row['rec_2'],
            view_2=row['view_2'],
        )
        
        return llm_client.get_response(prompt)

    def run_get_evidence(idx, row):
        """Wrapper function for parallel execution"""
        return idx, get_evidence(row)

    def parse_evidence(text):
        """Parse numbered evidence from text
        Format: 1. ~~~ [multiple lines] 2. ~~~ [multiple lines]
        """
        matches = re.findall(r'^\s*\d+\.\s*((?:.|\n)*?)(?=^\s*\d+\.|$)', str(text), re.MULTILINE)
        matches = [m.strip().replace('\n', ' ') for m in matches]  # Clean up newlines
        while len(matches) < 2:
            matches.append("")
        return matches[0], matches[1]

    # Initialize results
    results = [None] * len(data_df)
    
    # Parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_get_evidence, idx, row)
            for idx, row in data_df.iterrows()
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {outfile_name}"):
            idx, evidence = f.result()
            if idx < len(results):
                results[idx] = evidence

    # Add results to dataframe
    data_df = data_df.copy()
    data_df['evidence_str'] = results

    # Parse evidence without shuffling
    parsed = [parse_evidence(text) for text in results]
    data_df['evidence_1'] = [p[0] for p in parsed]
    data_df['evidence_2'] = [p[1] for p in parsed]

    # Select and save final columns
    result_df = data_df[["ticker", "evidence_str", "evidence_1", "evidence_2", "view_1", "view_2", "buy", "sell"]]
    result_df.to_csv(outfile_name, index=False)
    print(f"Saved to {outfile_name}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic evidence for stock recommendations")
    parser.add_argument("--output-dir", type=str, default="./data",
                       help="Output directory for generated files")
    parser.add_argument("--model-id", type=str, default=None,
                       help="Specific model ID (optional)")
    parser.add_argument("--temperature", type=float, default=0.6,
                       help="Temperature for generation")
    parser.add_argument("--max-workers", type=int, default=8,
                       help="Maximum number of concurrent workers")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit the number of items to process")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                       choices=["low", "medium", "high"],
                       help="Reasoning effort for reasoning models")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create client based on model choice
    client = LLMClient(model_id=args.model_id, temperature=args.temperature, reasoning_effort=args.reasoning_effort)
    
    print(f"Using model: {client.model_id}")
    
    # Prepare data
    current_df = expanded
    if args.limit:
        current_df = expanded.head(args.limit).copy()
        # Reset index to ensure 0..N indexing for results array
        current_df = current_df.reset_index(drop=True)
    
    output_path = os.path.join(args.output_dir, "evidence_corpus_view.csv")
    run_batch(BALANCED_CONFLICT_PROMPT_TEMPLATE, output_path, client, args.max_workers, current_df)