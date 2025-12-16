# Bash version

#

# MODEL_PATH="$HOME/projects/paligemma-weights/paligemma-3b-pt-224" # download https://huggingface.co/google/paligemma-3b-pt-224
# PARQUET_FILE="$HOME/Desktop/selection_image.parquet"
# IMAGES_FOLDER="$HOME/Desktop/images"
# OUTPUT_DIR="$HOME/projects/paligemma-weights/paligemma_lora"

# EPOCHS=1
# BATCH_SIZE=1
# ONLY_CPU="False"

# python finetune_paligemma_lora.py \
#     --model_path="$MODEL_PATH" \
#     --parquet_file="$PARQUET_FILE" \
#     --images_folder="$IMAGES_FOLDER" \
#     --output_dir="$OUTPUT_DIR" \
#     --epochs=$EPOCHS \
#     --batch_size=$BATCH_SIZE \
#     --only_cpu=$ONLY_CPU

# PowerShell version

# Ensure we have a clean start
$USERPROFILE = $env:USERPROFILE
$MODEL_PATH     = Join-Path $USERPROFILE "projects\paligemma-weights\paligemma-3b-pt-224"
$PARQUET_FILE   = Join-Path $USERPROFILE "Desktop\selection_image.parquet"
$IMAGES_FOLDER  = Join-Path $USERPROFILE "Desktop\images"
$OUTPUT_DIR     = Join-Path $USERPROFILE "projects\paligemma-weights\paligemma_lora"

# --- Other parameters ---
$EPOCHS     = 1
$BATCH_SIZE = 1
$ONLY_CPU   = "False"

# --- Verify required input paths exist ---
if (-not (Test-Path $MODEL_PATH)) { 
    Write-Error "MODEL_PATH not found: $MODEL_PATH"
    Read-Host "Press Enter to exit"
    exit 
}
if (-not (Test-Path $PARQUET_FILE)) { 
    Write-Error "PARQUET_FILE not found: $PARQUET_FILE"
    Read-Host "Press Enter to exit"
    exit 
}
if (-not (Test-Path $IMAGES_FOLDER)) { 
    Write-Error "IMAGES_FOLDER not found: $IMAGES_FOLDER"
    Read-Host "Press Enter to exit"
    exit 
}

# --- Create output directory if it doesn't exist ---
if (-not (Test-Path $OUTPUT_DIR)) {
    Write-Host "Creating output directory: $OUTPUT_DIR"
    New-Item -ItemType Directory -Path $OUTPUT_DIR -Force | Out-Null
}

# --- Display paths for verification ---
Write-Host "Using paths:"
Write-Host "  Model: $MODEL_PATH"
Write-Host "  Parquet: $PARQUET_FILE"  
Write-Host "  Images: $IMAGES_FOLDER"
Write-Host "  Output: $OUTPUT_DIR"
Write-Host ""

# --- Run Python script ---
python finetune_paligemma_lora.py `
    --model_path "$MODEL_PATH" `
    --parquet_file "$PARQUET_FILE" `
    --images_folder "$IMAGES_FOLDER" `
    --output_dir "$OUTPUT_DIR" `
    --epochs $EPOCHS `
    --batch_size $BATCH_SIZE `
    --only_cpu $ONLY_CPU

# --- Keep window open to see results ---
Read-Host "Press Enter to exit"