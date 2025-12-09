import os
import pandas as pd
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import argparse
from llm_clients import LLMClient, OpenAIClient, GeminiClient, TogetherClient, AnthropicClient, XAIClient

# Load data
df = pd.read_csv('./data/sp500_final.csv')
opinions = ['buy', 'sell']
expanded = df.loc[df.index.repeat(2)].copy()
expanded['opinion'] = opinions * len(df)  # Create opinion column by repeating
expanded = expanded.reset_index(drop=True)

# Prompt templates
QUAL_PROMPT_TEMPLATE = """
Stock Ticker: [{ticker}]
Stock Name: [{name}]
Recommendation: [{recommendation}]

Your task is to create a hypothetical investment case. Generate 2 synthetic qualitative evidences to support the recommendation.

Format your response strictly as:

Synthetic Qualitative Evidence:
1. ...
2. ...

Instructions:
- Only return the 2 structured qualitative evidences exactly as specified. Do not include any extra explanation, narrative, or titles.
- Create purely fictional evidences by inventing plausible company events that are not based on real-world public information.
- Use relative timeframes instead of specific calendar dates.
- Do not use placeholders or illustrative markers like "e.g." or "for example.".
- IMPORTANT: Each evidence must explicitly state a potential impact of exactly 5% on the stock price. This means:
  - For BUY recommendations: each evidence must explicitly state factors that could lead to a 5% price increase.
  - For SELL recommendations: each evidence must explicitly state factors that could lead to a 5% price decrease.
  - Avoid extreme language; use moderate terms like "modest growth", "slight decline", "moderate improvement", etc.
  - The cumulative effect of all evidences should reasonably justify about a 5% price movement
"""

QUANT_PROMPT_TEMPLATE = """
Stock Ticker: [{ticker}]
Stock Name: [{name}]
Recommendation: [{recommendation}]

Your task is to create a hypothetical investment case. Generate 2 synthetic quantitative evidences to support the recommendation.

Format your response strictly as:

Synthetic Quantitative Evidence:
1. ...
2. ...

Instructions:
- Only return the 2 structured quantitative evidences exactly as specified. Do not include any extra explanation, narrative, or titles.
- Create purely fictional evidences using specific numerical data, metrics, and financial figures.
- Each point must include concrete numbers, percentages, ratios, or other quantifiable metrics.
- Use relative timeframes instead of specific calendar dates.
- Do not use placeholders or illustrative markers like "e.g." or "for example.".
- IMPORTANT: Each evidence must use specific financial metrics to explicitly state a potential impact of exactly 5% on the stock price. This means:
  - For BUY recommendations: each evidence must present metrics that lead to a stated conclusion of a 5% price increase.
  - For SELL recommendations: each evidence must present metrics that lead to a stated conclusion of a 5% price decrease.
  - Use realistic financial metrics that align with a 5% valuation change
  - Avoid extreme numbers; use moderate figures that justify approximately 5% price movement
- Examples of quantitative evidence types:
  - Revenue/earnings growth rates
  - Profit margin changes
  - P/E ratio comparisons
  - Market share percentages
  - Cash flow metrics
  - Debt-to-equity ratios
  - Return on equity (ROE) figures
"""

MAX_WORKERS = 8

def run_batch(prompt_template, outfile_name, llm_client, max_workers, data_df):
    """Run batch processing for evidence generation"""
    
    def get_evidence(ticker, name, opinion):
        """Generate evidence for a single stock-opinion pair"""
        prompt = prompt_template.format(
            ticker=ticker,
            name=name,
            recommendation=opinion
        )
        return llm_client.get_response(prompt)

    def run_get_evidence(idx, row):
        """Wrapper function for parallel execution"""
        return idx, get_evidence(row['ticker'], row['name'], row['opinion'])

    def split_evidence(text):
        """Extract numbered evidence points from text"""
        matches = re.findall(r'\d+\.\s*(.*)', str(text))
        while len(matches) < 2:
            matches.append("")
        return matches[:2]

    # Initialize results list
    results = [None] * len(data_df)
    
    # Parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_get_evidence, idx, row)
            for idx, row in data_df.iterrows()
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {outfile_name}"):
            idx, evidence = f.result()
            # If data_df is a slice, idx might not start at 0 if not reset. 
            # But head() preserves index. If we want to be safe we should use enumerate or reset index.
            # Since we use head() on a reset_index dataframe, indices are 0..N.
            # However, to be safe if we pass a slice with non-zero start index:
            # We should probably map row index to result index or just use the row index if it matches.
            # For now assuming head() or reset_index() was used.
            if idx < len(results):
                results[idx] = evidence
            else:
                # Fallback if indices are messed up (shouldn't happen with head on 0-indexed df)
                pass

    # Process results
    data_df = data_df.copy() # Avoid SettingWithCopyWarning
    data_df['evidence_raw'] = results
    evidences = data_df['evidence_raw'].apply(split_evidence)
    evidence_df = pd.DataFrame(evidences.tolist(), columns=[f"evidence{i+1}" for i in range(2)])
    # Reset index of evidence_df to match data_df if needed, but they should align by length
    evidence_df.index = data_df.index
    
    result_df = pd.concat([data_df[['ticker', 'opinion']], evidence_df], axis=1)
    
    # Save to file
    result_df.to_csv(outfile_name, index=False)
    print(f"Saved to {outfile_name}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic evidence for stock recommendations")
    parser.add_argument("--type", type=str, required=True, 
                       choices=["qual", "quant", "both"],
                       help="Type of evidence to generate: qualitative, quantitative, or both")
    parser.add_argument("--output-dir", type=str, default="./data",
                       help="Output directory for generated files")
    parser.add_argument("--api", type=str, required=True, 
                       choices=["openai", "gemini", "together", "anthropic", "xai"],
                       help="Which API to use")
    parser.add_argument("--model-id", type=str, default=None,
                       help="Specific model ID (optional)")
    parser.add_argument("--temperature", type=float, default=0.6,
                       help="Temperature for generation")
    parser.add_argument("--max-workers", type=int, default=8,
                       help="Maximum number of concurrent workers")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit the number of items to process")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create client based on model choice
    if args.api == "openai":
        client = OpenAIClient(model_id=args.model_id, temperature=args.temperature)
    elif args.api == "gemini":
        client = GeminiClient(model_id=args.model_id, temperature=args.temperature)
    elif args.api == "together":
        client = TogetherClient(model_id=args.model_id, temperature=args.temperature)
    elif args.api == "anthropic":
        client = AnthropicClient(model_id=args.model_id, temperature=args.temperature)
    elif args.api == "xai":
        client = XAIClient(model_id=args.model_id, temperature=args.temperature)
    
    print(f"Using model: {client.model_id}")
    
    # Prepare data
    current_df = expanded
    if args.limit:
        current_df = expanded.head(args.limit).copy()
        # Reset index to ensure 0..N indexing for results array
        current_df = current_df.reset_index(drop=True)
    
    # Generate evidence based on type
    if args.type == "qual" or args.type == "both":
        qual_output = os.path.join(args.output_dir, "evidence_corpus_qual.csv")
        print("Generating qualitative evidence...")
        run_batch(QUAL_PROMPT_TEMPLATE, qual_output, client, args.max_workers, current_df)
    
    if args.type == "quant" or args.type == "both":
        quant_output = os.path.join(args.output_dir, "evidence_corpus_quant.csv")
        print("Generating quantitative evidence...")
        run_batch(QUANT_PROMPT_TEMPLATE, quant_output, client, args.max_workers, current_df)
    
    print("Evidence generation completed!")
