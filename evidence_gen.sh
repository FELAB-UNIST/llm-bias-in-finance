#!/bin/bash

# ==============================================================================
# Script to run evidence generation experiments for LLM bias testing.
# Supports volume and intensity evidence generation with flexible API/model selection.
# ===============================================================================

set -e

API_PROVIDER="gemini"  # Options: "openai", "gemini", "together", "anthropic", "xai"
MODEL_ID="gemini-2.5-pro"
TEMPERATURE=1.0
OUTPUT_DIR="./data"
MAX_WORKERS=30
EVIDENCE_TYPE="both"  # Options: "qual", "quant", "both"

# --- Volume Evidence Generation ---
echo "Running volume evidence generation..."
python evidence_generation_volume.py \
    --type $EVIDENCE_TYPE \
    --api $API_PROVIDER \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    --output-dir $OUTPUT_DIR \
    --max-workers $MAX_WORKERS \

echo "Volume evidence generation complete."

# --- Intensity Evidence Generation ---
echo "Running intensity evidence generation..."
python evidence_generation_intensity.py \
    --api $API_PROVIDER \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    --output-dir $OUTPUT_DIR \
    --max-workers $MAX_WORKERS \

# echo "Intensity evidence generation complete."

# You can add more experiment steps below as needed.

echo "All evidence generation experiments are complete."
