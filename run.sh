#!/bin/bash

# ==============================================================================
# Script to run LLM bias testing experiments.
#
# Parameters:
#   API_PROVIDER: The LLM provider to use. Options: "openai", "gemini", "together", "anthropic", "xai".
#   MODEL_ID: The specific model ID from the chosen provider.
#   TEMPERATURE: The temperature setting for the LLM's generation.
#   OUTPUT_DIR: The directory where result files will be saved.
#   MAX_WORKERS: The number of concurrent threads for API calls.
#   NUM_TRIALS: The number of trials to run for each stock in the volume experiment.
#   NUM_SETS: The number of experiment sets to run. Each set runs independently.
# ==============================================================================

set -e

# --- Configuration ---
API_PROVIDER="together"  # Options: "openai", "gemini", "together", "anthropic", "xai"
MODEL_ID="DeepSeek-V3"
TEMPERATURE=0.6
OUTPUT_DIR="./exp_result"
MAX_WORKERS=40
NUM_TRIALS=10
NUM_SETS=3

# --- Experiment 1: Attribute Preference Test ---
# This experiment tests if the LLM shows a preference for certain stock attributes (e.g., sector, market cap)
# when presented with an equal number of buy and sell arguments.
# Runs the attribute preference experiment.
python bias_attribute.py \
    --api $API_PROVIDER \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    --output-dir $OUTPUT_DIR \
    --max-workers $MAX_WORKERS \
    --num-trials $NUM_TRIALS \
    --num-sets $NUM_SETS

# Analyzes the results from the attribute preference experiment.
python result_attribute.py \
    --model-id $MODEL_ID \
    --output-dir $OUTPUT_DIR

# --- Experiment 2: Strategy Preference Test ---
# This experiment tests if the LLM prefers a "momentum" or "contrarian" investment strategy.
# Runs the strategy preference experiment.
# python bias_strategy.py \
#     --api $API_PROVIDER \
#     --model-id $MODEL_ID \
#     --temperature $TEMPERATURE \
#     --output-dir $OUTPUT_DIR \
#     --max-workers $MAX_WORKERS \
#     --num-sets $NUM_SETS

# # Analyzes the results from the strategy preference experiment.
# python result_strategy.py \
#     --model-id $MODEL_ID \
#     --output-dir $OUTPUT_DIR

echo "All experiments and analyses are complete."