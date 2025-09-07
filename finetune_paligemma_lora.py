import os
import gc
import glob
import fire
import torch
import pandas as pd
from PIL import Image
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn

from utils import load_hf_model  # returns (model, tokenizer)

# -------------------------
# Dataset validation utility
# -------------------------
def validate_dataset(parquet_file: str, images_folder: str, max_check: int = 100):
    """Pre-validate dataset to catch corrupted images early"""
    print("Validating dataset...")
    df = pd.read_parquet(parquet_file)
    corrupted_files = []
    missing_files = []
    
    check_count = min(len(df), max_check)
    for i in range(check_count):
        row = df.iloc[i]
        source_id = str(row["source_identifier"])
        page_idx = int(row["FEATURE_page_indexes"])
        
        # Find image file
        patterns = [
            os.path.join(images_folder, f"{source_id}_p{page_idx}.*"),
            os.path.join(images_folder, f"{source_id}_{page_idx}.*"),
            os.path.join(images_folder, f"{source_id}*{page_idx}.*"),
        ]
        
        image_path = None
        for pat in patterns:
            matches = glob.glob(pat)
            if matches:
                image_path = matches[0]
                break
        
        if image_path is None:
            missing_files.append(f"{source_id}_p{page_idx}")
            continue
            
        # Validate image
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            corrupted_files.append((image_path, str(e)))
    
    print(f"Validated {check_count} samples:")
    print(f"  Missing files: {len(missing_files)}")
    print(f"  Corrupted files: {len(corrupted_files)}")
    
    if corrupted_files:
        print("Corrupted files found:")
        for path, error in corrupted_files[:5]:  # Show first 5
            print(f"  {path}: {error}")
    
    return len(corrupted_files) == 0 and len(missing_files) == 0

# -------------------------
# Dataset
# -------------------------
class FinancialImageDataset(Dataset):
    def __init__(self, parquet_file: str, images_folder: str, tokenizer, image_size: int = 224, max_length: int = 512, max_samples: Optional[int] = None):
        self.df = pd.read_parquet(parquet_file)
        if max_samples:
            self.df = self.df.head(max_samples)
            print(f"Limited dataset to {len(self.df)} samples for faster training")
        self.images_folder = images_folder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def _find_image(self, source_id: str, page_idx: int):
        patterns = [
            os.path.join(self.images_folder, f"{source_id}_p{page_idx}.*"),
            os.path.join(self.images_folder, f"{source_id}_{page_idx}.*"),
            os.path.join(self.images_folder, f"{source_id}*{page_idx}.*"),
        ]
        for pat in patterns:
            matches = glob.glob(pat)
            if matches:
                return matches[0]
        return None

    def __getitem__(self, idx):
        max_retries = 10  # Try up to 10 different samples if images are corrupted
        for attempt in range(max_retries):
            try:
                current_idx = (idx + attempt) % len(self.df)  # Cycle through dataset
                row = self.df.iloc[current_idx]
                source_id = str(row["source_identifier"])
                page_idx = int(row["FEATURE_page_indexes"])
                image_path = self._find_image(source_id, page_idx)
                
                if image_path is None:
                    print(f"Warning: No image found for source={source_id} page={page_idx}, trying next sample...")
                    continue

                # Load and process image with robust error handling
                try:
                    with Image.open(image_path) as img:
                        # Verify image is valid by loading it
                        img.verify()
                    
                    # Reopen after verify (verify closes the file)
                    with Image.open(image_path) as img:
                        image = img.convert("RGB")
                        pixel_values = self.transform(image)
                        
                except (OSError, IOError, Image.UnidentifiedImageError) as e:
                    print(f"Warning: Corrupted image {image_path}: {e}, trying next sample...")
                    continue

                text = str(row.get("FEATURE_full_prompt", row.get("template_id", "")))
                tokenized = self.tokenizer(
                    text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
                )
                input_ids = tokenized["input_ids"].squeeze(0)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

                return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}
                
            except Exception as e:
                print(f"Warning: Error processing sample {current_idx}: {e}, trying next sample...")
                continue
        
        # If we get here, all retries failed - create a dummy sample to keep training going
        print(f"Error: Could not load any valid sample around index {idx}, using dummy data")
        dummy_image = Image.new('RGB', (224, 224), color='white')
        pixel_values = self.transform(dummy_image)
        
        dummy_text = "dummy text"
        tokenized = self.tokenizer(
            dummy_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}


# -------------------------
# Memory management utilities
# -------------------------
def clear_memory(device):
    """Clear memory based on device type"""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device == "mps":
        try:
            torch.mps.empty_cache()
            torch.mps.synchronize()
        except RuntimeError as e:
            if "watermark" in str(e).lower():
                print(f"MPS cache clear warning: {e}")
            else:
                raise e

def get_memory_usage(device):
    """Get current memory usage"""
    if device == "cuda":
        return torch.cuda.memory_allocated() / 1024**3  # GB
    elif device == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1024**3  # GB
        except RuntimeError:
            return 0  # Return 0 if MPS memory tracking unavailable
    return 0

# -------------------------
# Patch model for PEFT
# -------------------------
def patch_model_for_peft(model):
    import inspect
    if not hasattr(model, "config"):
        raise RuntimeError("Model has no .config attribute")
    if not hasattr(model.config, "model_type"):
        model.config.model_type = "causal_lm"
    if not hasattr(model.config, "get"):
        model.config.get = lambda k, default=None: getattr(model.config, k, default)
    if not hasattr(model, "prepare_inputs_for_generation"):
        def _prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}
        model.prepare_inputs_for_generation = _prepare_inputs_for_generation.__get__(model, model.__class__)
    if not hasattr(model, "get_output_embeddings"):
        def _get_output_embeddings(self):
            if hasattr(model, "lm_head"):
                return model.lm_head
            if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
                return model.language_model.lm_head
            return None
        model.get_output_embeddings = _get_output_embeddings.__get__(model, model.__class__)

    # Wrap forward
    base_model = getattr(model, "base_model", model)
    original_forward = base_model.forward
    def forward_wrapper(*args, **kwargs):
        sig = inspect.signature(original_forward)
        allowed_args = sig.parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_args}
        if "input_ids" in filtered_kwargs and "attention_mask" not in filtered_kwargs:
            filtered_kwargs["attention_mask"] = torch.ones_like(filtered_kwargs["input_ids"])
        return original_forward(*args, **filtered_kwargs)
    base_model.forward = forward_wrapper


# -------------------------
# FIXED: Robust checkpoint saving function
# -------------------------
def save_checkpoint_robust(model, tokenizer, output_dir, step_info=""):
    """Robust checkpoint saving with multiple fallback methods"""
    import json
    from pathlib import Path
    
    success = False
    error_msgs = []
    
    try:
        # Ensure directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Method 1: Try standard PEFT save_pretrained
        try:
            model.save_pretrained(output_dir)
            print(f"✓ Method 1 success: Saved LoRA adapter to {output_dir}")
            success = True
        except Exception as e1:
            error_msgs.append(f"Method 1 (save_pretrained): {str(e1)}")
            
            # Method 2: Try saving just the adapter state dict
            try:
                # Get PEFT model state dict
                if hasattr(model, 'peft_config'):
                    state_dict = model.state_dict()
                    # Filter to only LoRA parameters
                    adapter_state_dict = {k: v for k, v in state_dict.items() if 'lora_' in k.lower()}
                    
                    if adapter_state_dict:
                        torch.save(adapter_state_dict, os.path.join(output_dir, "adapter_model.bin"))
                        
                        # Save adapter config
                        config_dict = model.peft_config['default'].to_dict()
                        with open(os.path.join(output_dir, "adapter_config.json"), 'w') as f:
                            json.dump(config_dict, f, indent=2)
                        
                        print(f"✓ Method 2 success: Saved adapter weights and config to {output_dir}")
                        success = True
                    else:
                        raise ValueError("No LoRA parameters found in state dict")
                        
            except Exception as e2:
                error_msgs.append(f"Method 2 (manual adapter save): {str(e2)}")
                
                # Method 3: Save full model state dict as last resort
                try:
                    full_state_dict = model.state_dict()
                    torch.save(full_state_dict, os.path.join(output_dir, "full_model_state.bin"))
                    print(f"✓ Method 3 success: Saved full model state to {output_dir}")
                    success = True
                except Exception as e3:
                    error_msgs.append(f"Method 3 (full state dict): {str(e3)}")
    
        # Save tokenizer (this usually works)
        try:
            if tokenizer:
                tokenizer.save_pretrained(output_dir)
                print(f"✓ Saved tokenizer to {output_dir}")
        except Exception as e:
            error_msgs.append(f"Tokenizer save: {str(e)}")
        
        # Save training metadata
        try:
            metadata = {
                "step_info": step_info,
                "timestamp": torch.cuda.Event().record() if torch.cuda.is_available() else "unknown",
                "success": success,
                "errors": error_msgs if error_msgs else None
            }
            
            with open(os.path.join(output_dir, "checkpoint_info.json"), 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
            
    except Exception as e:
        error_msgs.append(f"Directory creation: {str(e)}")
        print(f"✗ Failed to create checkpoint directory: {e}")
        
    if success:
        print(f"🎉 Checkpoint saved successfully to {output_dir}")
    else:
        print(f"💥 All checkpoint save methods failed:")
        for i, msg in enumerate(error_msgs, 1):
            print(f"   {i}. {msg}")
            
    return success


# -------------------------
# Training function (your original logic + fixed checkpoints)
# -------------------------
def train(
    model_path: str,
    parquet_file: str,
    images_folder: str,
    output_dir: str = "paligemma_lora",
    epochs: int = 1,  # Reduced from 3 for faster testing
    batch_size: int = 1,
    lr: float = 1e-4,
    max_length: int = 256,  # Reduced from 512 for speed
    only_cpu: bool = False,
    image_size: Optional[int] = 224,  # Keep your original size
    device: Optional[str] = None,
    accum_steps: int = 8,  # Increased for effective batch size
    save_every_n_steps: int = 50,  # More frequent saves
    max_memory_gb: float = 15.0,
    max_samples: Optional[int] = 200,  # NEW: Limit samples for testing
):
    # Device selection
    if device is None:
        device = "cpu"
        if not only_cpu:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
    print("Device:", device)
    
    # Set memory management for MPS - be more careful with environment variables
    if device == "mps":
        # Only set if not already set, and use a safer value
        if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable upper limit as suggested in error
        print("Configured MPS memory management")

    # Clear initial memory
    clear_memory(device)
    
    # Validate dataset first to catch issues early
    print("Running dataset validation...")
    is_valid = validate_dataset(parquet_file, images_folder, max_check=min(100, max_samples or 100))
    if not is_valid:
        print("Warning: Dataset validation found issues, but continuing with robust error handling...")

    # Load model + tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_hf_model(model_path, device)
    
    # Move to device and clear memory
    model = model.to(device)
    clear_memory(device)
    
    print(f"Initial memory usage: {get_memory_usage(device):.2f} GB")

    # Image size
    if image_size is None and hasattr(model.config, "vision_config") and hasattr(model.config.vision_config, "image_size"):
        try:
            image_size = int(model.config.vision_config.image_size)
        except Exception:
            image_size = 224
    image_size = image_size or 224
    print(f"Using image size: {image_size}")

    # Patch + PEFT with more conservative settings
    print("Setting up LoRA...")
    patch_model_for_peft(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=8,  # Reduced from 16 to save memory
        lora_alpha=16,  # Adjusted accordingly
        lora_dropout=0.1, 
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config).to(device)
    model.train()

    # Enable gradient checkpointing if available
    base_model = getattr(model, "base_model", model)
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")
    else:
        print("Gradient checkpointing not supported for this model, skipping.")

    clear_memory(device)
    print(f"Memory after model setup: {get_memory_usage(device):.2f} GB")

    # Dataset + DataLoader with smaller batch processing
    print("Loading dataset...")
    dataset = FinancialImageDataset(
        parquet_file, images_folder, tokenizer, 
        image_size=image_size, max_length=max_length, max_samples=max_samples
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Disable multiprocessing to save memory
        pin_memory=False  # Disable pin_memory to save memory
    )
    print(f"Dataset size: {len(dataset)} samples")

    # Optimizer + Loss
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, foreach=False)
    ignore_index = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else -100
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)

    # GradScaler with updated API
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    print("Starting training...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Accumulation steps: {accum_steps}")
    print(f"Max memory limit: {max_memory_gb} GB")
    
    for epoch in range(epochs):
        running_loss = 0.0
        steps = 0
        optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            # Check memory before processing
            current_memory = get_memory_usage(device)
            if current_memory > max_memory_gb:
                print(f"Warning: Memory usage {current_memory:.2f} GB exceeds limit {max_memory_gb} GB")
                clear_memory(device)
                current_memory = get_memory_usage(device)
                print(f"After cleanup: {current_memory:.2f} GB")

            try:
                # Move batch to device
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)

                # Match dtype and optimize memory
                model_dtype = next(model.parameters()).dtype
                pixel_values = pixel_values.to(dtype=model_dtype)

                # Use appropriate autocast based on device
                if device == "cuda":
                    autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
                elif device == "mps":
                    autocast_ctx = torch.autocast("cpu", dtype=torch.float32)  # MPS autocast can be unstable
                else:
                    autocast_ctx = torch.autocast("cpu", dtype=torch.float32)

                with autocast_ctx:
                    outputs = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        pixel_values=pixel_values, 
                        return_dict=True
                    )
                    logits = outputs.get("logits")
                    if logits is None:
                        raise RuntimeError("Model did not return logits")

                    # Compute loss with memory optimization
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()

                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss / accum_steps

                # Backward pass with error handling
                if device == "mps":
                    # For MPS, disable autocast during backward to avoid memory issues
                    loss.backward()
                else:
                    scaler.scale(loss).backward()

                # Gradient accumulation
                if (i + 1) % accum_steps == 0:
                    if device == "mps":
                        # Clip gradients to prevent explosion
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad()

                running_loss += loss.item() * accum_steps
                steps += 1

                # Aggressive memory cleanup
                del outputs, logits, shift_logits, shift_labels, loss
                del input_ids, attention_mask, pixel_values
                
                # More conservative memory cleanup
                if device == "mps":
                    gc.collect()  # Only use gc for MPS to avoid watermark issues
                elif device == "cuda":
                    clear_memory(device)

                # Progress reporting
                if (i + 1) % 10 == 0:
                    current_loss = running_loss / max(1, steps)
                    current_memory = get_memory_usage(device)
                    print(f"Epoch {epoch+1}/{epochs}, Step {i+1}, Loss: {current_loss:.6f}, Memory: {current_memory:.2f} GB")

                # FIXED: Robust checkpoint saving
                if save_every_n_steps > 0 and (i + 1) % save_every_n_steps == 0:
                    checkpoint_dir = f"{output_dir}_checkpoint_epoch{epoch+1}_step{i+1}"
                    print(f"Saving checkpoint to {checkpoint_dir}...")
                    save_checkpoint_robust(model, tokenizer, checkpoint_dir, f"epoch_{epoch+1}_step_{i+1}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM error at step {i+1}: {e}")
                    print("Attempting memory cleanup...")
                    
                    # Aggressive cleanup
                    if 'outputs' in locals(): del outputs
                    if 'logits' in locals(): del logits
                    if 'loss' in locals(): del loss
                    if 'input_ids' in locals(): del input_ids
                    if 'attention_mask' in locals(): del attention_mask
                    if 'pixel_values' in locals(): del pixel_values
                    
                    clear_memory(device)
                    
                    # Reset gradients
                    optimizer.zero_grad()
                    
                    print(f"Memory after cleanup: {get_memory_usage(device):.2f} GB")
                    print("Skipping this batch and continuing...")
                    continue
                else:
                    raise e

        # End of epoch cleanup
        clear_memory(device)
        avg_loss = running_loss / max(1, steps)
        final_memory = get_memory_usage(device)
        print(f"Epoch {epoch+1}/{epochs} completed - avg_loss: {avg_loss:.6f}, Memory: {final_memory:.2f} GB")

    # FIXED: Final save with robust error handling
    print("Saving final model...")
    final_success = save_checkpoint_robust(model, tokenizer, output_dir, "final_model")
    
    if final_success:
        print(f"🎉 Training completed successfully! Final model saved to {output_dir}")
    else:
        print(f"⚠️ Training completed but final save had issues. Check {output_dir} for partial saves.")

    # Final cleanup
    clear_memory(device)
    print("Training completed!")


if __name__ == "__main__":
    fire.Fire(train)