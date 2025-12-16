import os
from pathlib import Path
from finetune_paligemma_lora import train

# Get user profile path
USERPROFILE = os.environ.get('USERPROFILE')
if not USERPROFILE:
    raise ValueError("USERPROFILE environment variable not found")

# Define paths
MODEL_PATH = os.path.join(USERPROFILE, "projects", "paligemma-weights", "paligemma-3b-pt-224")
PARQUET_FILE = os.path.join(USERPROFILE, "Desktop", "selection_image.parquet")
IMAGES_FOLDER = os.path.join(USERPROFILE, "Desktop", "images")
OUTPUT_DIR = os.path.join(USERPROFILE, "projects", "paligemma-weights", "paligemma_lora")

# Training parameters
EPOCHS = 1
BATCH_SIZE = 1
ONLY_CPU = False  # Use boolean instead of string

# Verify required input paths exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

if not os.path.exists(PARQUET_FILE):
    raise FileNotFoundError(f"PARQUET_FILE not found: {PARQUET_FILE}")

if not os.path.exists(IMAGES_FOLDER):
    raise FileNotFoundError(f"IMAGES_FOLDER not found: {IMAGES_FOLDER}")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# Display paths for verification
print("Using paths:")
print(f"  Model: {MODEL_PATH}")
print(f"  Parquet: {PARQUET_FILE}")
print(f"  Images: {IMAGES_FOLDER}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  CPU Only: {ONLY_CPU}")
print()

# Run training
try:
    train(MODEL_PATH, PARQUET_FILE, IMAGES_FOLDER, OUTPUT_DIR, EPOCHS, BATCH_SIZE, ONLY_CPU)
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {e}")
    raise

print("Done!")
input("Press Enter to exit...")