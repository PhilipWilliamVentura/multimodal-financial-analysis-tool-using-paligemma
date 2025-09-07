#!/bin/bash

MODEL_PATH="$HOME/projects/paligemma-weights/paligemma-3b-pt-224" # download https://huggingface.co/google/paligemma-3b-pt-224
PARQUET_FILE="$HOME/Desktop/selection_image.parquet"
IMAGES_FOLDER="$HOME/Desktop/images"
OUTPUT_DIR="$HOME/projects/paligemma-weights/paligemma_lora"

EPOCHS=1
BATCH_SIZE=1
ONLY_CPU="False"

python finetune_paligemma_lora.py \
    --model_path="$MODEL_PATH" \
    --parquet_file="$PARQUET_FILE" \
    --images_folder="$IMAGES_FOLDER" \
    --output_dir="$OUTPUT_DIR" \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --only_cpu=$ONLY_CPU
