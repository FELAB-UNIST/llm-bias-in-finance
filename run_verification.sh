#!/bin/bash

# ==============================================================================
# Script to run LLM bias verification experiments (Volume Test).
#
# Parameters:
#   MODEL_ID: OpenRouter model ID (e.g., "openai/gpt-4.1", "anthropic/claude-sonnet-4")
#   TEMPERATURE: The temperature setting for the LLM's generation.
#   REASONING_EFFORT: Reasoning effort level ("low", "medium", "high") for reasoning models. Leave empty for non-reasoning models.
#   OUTPUT_DIR: The directory where result files will be saved.
#   MAX_WORKERS: The number of concurrent threads for API calls.
#   NUM_TRIALS: The number of trials to run for each stock in the volume experiment.
# ==============================================================================

set -e

# --- Configuration ---
MODEL_ID="deepseek/deepseek-chat-v3-0324"  # OpenRouter model ID
REASONING_EFFORT=""  # "low", "medium", "high" for reasoning models, empty for regular models
INPUT_DIR="./mixed_result"
OUTPUT_DIR="./bias_verification"
MAX_WORKERS=40
NUM_TRIALS=1
TEMPERATURE=0.6
MAX_TOKENS="1024"  # Maximum tokens for response, empty for model default

# Build reasoning effort argument if set
REASONING_ARG=""
# Extract short model ID (everything after the last /)
MODEL_SUFFIX=$(echo $MODEL_ID | awk -F/ '{print $NF}')

if [ -n "$REASONING_EFFORT" ]; then
    REASONING_ARG="--reasoning-effort $REASONING_EFFORT"
    MODEL_SUFFIX="${MODEL_SUFFIX}_${REASONING_EFFORT}"
fi

# Build max tokens argument if set (only if reasoning effort is not set)
MAX_TOKENS_ARG=""
if [ -n "$MAX_TOKENS" ] && [ -z "$REASONING_EFFORT" ]; then
    MAX_TOKENS_ARG="--max-tokens $MAX_TOKENS"
fi

# Construct JSON path based on model ID and output directory
# This assumes result_attribute.py has been run and generated the JSON file
JSON_PATH="${INPUT_DIR}/${MODEL_SUFFIX}_att_result.json"

echo "Running verification for model: $MODEL_ID"
echo "Using JSON results from: $JSON_PATH"

if [ ! -f "$JSON_PATH" ]; then
    echo "Error: JSON file not found at $JSON_PATH"
    echo "Please run the attribute preference test first (run.sh) to generate the results."
    exit 1
fi

# --- Experiment: Verification Volume Test ---
# This experiment verifies bias by varying the volume of evidence for high-bias groups.
python bias_verification_score.py \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    $REASONING_ARG \
    $MAX_TOKENS_ARG \
    --output-dir $OUTPUT_DIR \
    --input-dir $INPUT_DIR \
    --max-workers $MAX_WORKERS \
    --num-trials $NUM_TRIALS \
    --json-path $JSON_PATH

echo "Verification experiment complete."
