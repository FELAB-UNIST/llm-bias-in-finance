#!/bin/bash

# ==============================================================================
# Script to run LLM bias testing experiments.
#
# Parameters:
#   MODEL_ID: OpenRouter model ID (e.g., "openai/gpt-4.1", "anthropic/claude-sonnet-4")
#   TEMPERATURE: The temperature setting for the LLM's generation.
#   REASONING_EFFORT: Reasoning effort level ("low", "medium", "high") for reasoning models. Leave empty for non-reasoning models.
#   OUTPUT_DIR: The directory where result files will be saved.
#   MAX_WORKERS: The number of concurrent threads for API calls.
#   NUM_TRIALS: The number of trials to run for each stock in the volume experiment.
#   NUM_SETS: The number of experiment sets to run. Each set runs independently.
# ==============================================================================

set -e

# --- Configuration ---
MODEL_ID="meta-llama/llama-4-maverick"  # OpenRouter model ID
REASONING_EFFORT=""  # "low", "medium", "high" for reasoning models, empty for regular models
OUTPUT_DIR="./exp_result"
MAX_WORKERS=40
NUM_TRIALS=10
NUM_SETS=2
TEMPERATURE=0.6
MAX_TOKENS="1024"  # Maximum tokens for response, empty for model default

# Build reasoning effort argument if set
REASONING_ARG=""
if [ -n "$REASONING_EFFORT" ]; then
    REASONING_ARG="--reasoning-effort $REASONING_EFFORT"
fi

# Build max tokens argument if set (only if reasoning effort is not set)
MAX_TOKENS_ARG=""
if [ -n "$MAX_TOKENS" ] && [ -z "$REASONING_EFFORT" ]; then
    MAX_TOKENS_ARG="--max-tokens $MAX_TOKENS"
fi


# --- Experiment 1: Attribute Preference Test ---
# This experiment tests if the LLM shows a preference for certain stock attributes (e.g., sector, market cap)
# when presented with an equal number of buy and sell arguments.
# Runs the attribute preference experiment.

python bias_attribute.py \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    $REASONING_ARG \
    $MAX_TOKENS_ARG \
    --output-dir $OUTPUT_DIR \
    --max-workers $MAX_WORKERS \
    --num-trials $NUM_TRIALS \
    --num-sets $NUM_SETS

# Analyzes the results from the attribute preference experiment.
python result_attribute.py \
    --model-id $MODEL_ID \
    $REASONING_ARG \
    --output-dir $OUTPUT_DIR

# --- Experiment 2: Strategy Preference Test ---
# This experiment tests if the LLM prefers a "momentum" or "contrarian" investment strategy.
# Runs the strategy preference experiment.
# python bias_strategy.py \
#     --model-id $MODEL_ID \
#     --temperature $TEMPERATURE \
#     $REASONING_ARG \
#     --output-dir $OUTPUT_DIR \
#     --max-workers $MAX_WORKERS \
#     --num-sets $NUM_SETS

# # Analyzes the results from the strategy preference experiment.
# python result_strategy.py \
#     --model-id $MODEL_ID \
#     $REASONING_ARG \
#     --output-dir $OUTPUT_DIR

echo "All experiments and analyses are complete."