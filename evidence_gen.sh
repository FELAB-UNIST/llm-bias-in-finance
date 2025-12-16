#!/bin/bash

# ==============================================================================
# Script to run evidence generation experiments for LLM bias testing.
# Uses OpenRouter API for all models.
# ===============================================================================

set -e

# --- Configuration ---
MODEL_ID="google/gemini-2.5-pro"  # OpenRouter model ID
TEMPERATURE=1.0
REASONING_EFFORT=""  # "low", "medium", "high" for reasoning models, empty for regular models
OUTPUT_DIR="./data_gemini"
MAX_WORKERS=30
EVIDENCE_TYPE="quant"  # Options: "qual", "quant", "both"

# Build reasoning effort argument if set
REASONING_ARG=""
if [ -n "$REASONING_EFFORT" ]; then
    REASONING_ARG="--reasoning-effort $REASONING_EFFORT"
fi

# --- Volume Evidence Generation ---
echo "Running volume evidence generation..."
python evidence_generation_volume.py \
    --type $EVIDENCE_TYPE \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    $REASONING_ARG \
    --output-dir $OUTPUT_DIR \
    --max-workers $MAX_WORKERS

# echo "Volume evidence generation complete."

# --- Intensity Evidence Generation ---
echo "Running intensity evidence generation..."
python evidence_generation_intensity.py \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    $REASONING_ARG \
    --output-dir $OUTPUT_DIR \
    --max-workers $MAX_WORKERS

# echo "Intensity evidence generation complete."

# You can add more experiment steps below as needed.

echo "All evidence generation experiments are complete."
