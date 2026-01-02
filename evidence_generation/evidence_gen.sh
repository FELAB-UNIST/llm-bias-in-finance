#!/bin/bash

# ==============================================================================
# Script to run evidence generation experiments for LLM bias testing.
# Supports volume and intensity evidence generation with flexible API/model selection.
# ===============================================================================

set -e

MODEL_ID="minimax/minimax-m2"
TEMPERATURE=1.0
OUTPUT_DIR="./data/mini"
MAX_WORKERS=30
EVIDENCE_TYPE="quant"  # Options: "qual", "quant", "both"
REASONING_EFFORT="high"  # Options: "low", "medium", "high" (only for reasoning models)

# --- Volume Evidence Generation ---
echo "Running volume evidence generation..."

REASONING_ARG=""
if [ -n "$REASONING_EFFORT" ]; then
    REASONING_ARG="--reasoning-effort $REASONING_EFFORT"
fi

python evidence_generation_volume.py \
    --type $EVIDENCE_TYPE \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    --output-dir $OUTPUT_DIR \
    --max-workers $MAX_WORKERS \
    $REASONING_ARG \

echo "Volume evidence generation complete."

# --- Intensity Evidence Generation ---
# echo "Running intensity evidence generation..."
# python evidence_generation_intensity.py \
#     --model-id $MODEL_ID \
#     --temperature $TEMPERATURE \
#     --output-dir $OUTPUT_DIR \
#     --max-workers $MAX_WORKERS \

# echo "Intensity evidence generation complete."

# You can add more experiment steps below as needed.

echo "All evidence generation experiments are complete."
