# ablation_study.py
# COMPLETE ABLATION STUDY - DO NOT MODIFY YOUR EXISTING FILES
# Just run: python ablation_study.py

import torch
import time
import psutil
import json
import os
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from datetime import datetime

# Import from your existing files
from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

MODEL_PATH = "/Users/Philip Ventura/projects/paligemma-weights/paligemma-3b-pt-224"  # CHANGE THIS TO YOUR PATH
OUTPUT_DIR = "ablation_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test configurations
CONFIGS = [
    {"name": "baseline", "kv_cache": True, "dtype": torch.float16, "temperature": 0.8, "max_tokens": 100},
    {"name": "no_kv_cache", "kv_cache": False, "dtype": torch.float16, "temperature": 0.8, "max_tokens": 100},
    {"name": "fp32", "kv_cache": True, "dtype": torch.float32, "temperature": 0.8, "max_tokens": 100},
    {"name": "temp_0", "kv_cache": True, "dtype": torch.float16, "temperature": 0.0, "max_tokens": 100},
    {"name": "temp_1", "kv_cache": True, "dtype": torch.float16, "temperature": 1.0, "max_tokens": 100},
    {"name": "short_gen", "kv_cache": True, "dtype": torch.float16, "temperature": 0.8, "max_tokens": 50},
    {"name": "long_gen", "kv_cache": True, "dtype": torch.float16, "temperature": 0.8, "max_tokens": 200},
]

# Small benchmark dataset (will download automatically)
BENCHMARK = [
    {
        "url": "https://images.unsplash.com/photo-1574158622682-e40e69881006",
        "prompt": "What animal is this?",
        "expected": "cat"
    },
    {
        "url": "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e",
        "prompt": "What color is the car?",
        "expected": "red"
    },
    {
        "url": "https://images.unsplash.com/photo-1551963831-b3b1ca40c98e",
        "prompt": "What food is shown?",
        "expected": "breakfast"
    },
    {
        "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
        "prompt": "Describe the landscape",
        "expected": "mountains"
    },
    {
        "url": "https://images.unsplash.com/photo-1511367461989-f85a21fda167",
        "prompt": "What is the main subject?",
        "expected": "profile"
    },
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def download_image(url, save_path):
    """Download image from URL"""
    try:
        response = requests.get(url + "?w=400", timeout=10)
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img.save(save_path)
        return img
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # Create a dummy image if download fails
        img = Image.new('RGB', (224, 224), color='red')
        img.save(save_path)
        return img

def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def move_inputs_to_device(model_inputs, device):
    """Move model inputs to device"""
    return {k: v.to(device) for k, v in model_inputs.items()}

def _sample_top_p(probs, p):
    """Top-p sampling"""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def run_inference(model, processor, image_path, prompt, config):
    """
    Run inference with specific configuration
    Returns: (output_text, latency_ms, tokens_generated, memory_mb)
    """
    # Prepare inputs
    image = Image.open(image_path)
    model_inputs = processor(text=[prompt], images=[image])
    model_inputs = move_inputs_to_device(model_inputs, DEVICE)
    
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    
    # Convert model to specified dtype
    model = model.to(config["dtype"])
    pixel_values = pixel_values.to(config["dtype"])
    
    # Initialize KV cache if needed
    kv_cache = KVCache() if config["kv_cache"] else None
    
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    
    # Measure memory before
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mem_before = get_gpu_memory()
    
    # Start timing
    start_time = time.time()
    
    # Generation loop
    for _ in range(config["max_tokens"]):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache
            )
        
        if config["kv_cache"]:
            kv_cache = outputs["kv_cache"]
        
        next_token_logits = outputs["logits"][:, -1, :]
        
        # Sampling based on temperature
        if config["temperature"] == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = torch.softmax(next_token_logits / config["temperature"], dim=-1)
            next_token = _sample_top_p(next_token_logits, p=0.9)
        
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)
        
        if next_token.item() == stop_token:
            break
        
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )
        
        # Only pass pixel_values on first iteration
        pixel_values = None
    
    # End timing
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    # Measure memory after
    mem_after = get_gpu_memory()
    memory_used = mem_after - mem_before
    
    # Decode output
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return decoded, latency_ms, len(generated_tokens), memory_used

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("="*80)
    print("PALIGEMMA ABLATION STUDY")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    
    # Download benchmark images
    print("Step 1: Downloading benchmark images...")
    for i, item in enumerate(BENCHMARK):
        save_path = f"{OUTPUT_DIR}/images/img_{i}.jpg"
        if not os.path.exists(save_path):
            print(f"  Downloading image {i+1}/{len(BENCHMARK)}...")
            download_image(item["url"], save_path)
        item["image_path"] = save_path
    print("✓ Images ready\n")
    
    # Load model - SIMPLE METHOD (no accelerate)
    print("Step 2: Loading model...")
    print("  Loading config...")
    
    import json
    from transformers import AutoTokenizer
    from modeling_gemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration
    
    # Load config
    with open(os.path.join(MODEL_PATH, "config.json"), "r") as f:
        config_dict = json.load(f)
    config = PaliGemmaConfig(**config_dict)
    
    # Create model
    print("  Creating model...")
    model = PaliGemmaForConditionalGeneration(config)
    
    # Load weights manually (slower but reliable)
    print("  Loading weights (this may take 2-3 minutes)...")
    import safetensors.torch
    
    # Find all safetensor files
    weight_files = list(Path(MODEL_PATH).glob("*.safetensors"))
    if not weight_files:
        weight_files = list(Path(MODEL_PATH).glob("*.bin"))
    
    print(f"  Found {len(weight_files)} weight files")
    
    # Load all weights
    state_dict = {}
    for weight_file in weight_files:
        print(f"    Loading {weight_file.name}...")
        if str(weight_file).endswith(".safetensors"):
            weights = safetensors.torch.load_file(str(weight_file))
        else:
            weights = torch.load(str(weight_file), map_location="cpu")
        state_dict.update(weights)
    
    print("  Loading state dict into model...")
    model.load_state_dict(state_dict, strict=False)
    
    # Move to device
    print(f"  Moving model to {DEVICE}...")
    model = model.to(DEVICE)
    model.eval()
    
    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
    print("✓ Model loaded\n")
    
    # Run experiments
    print("Step 3: Running ablation experiments...")
    print(f"Total experiments: {len(CONFIGS)} configs × {len(BENCHMARK)} images = {len(CONFIGS) * len(BENCHMARK)}")
    print()
    
    results = []
    
    for config_idx, config in enumerate(CONFIGS):
        print(f"[{config_idx+1}/{len(CONFIGS)}] Testing configuration: {config['name']}")
        print(f"  Settings: kv_cache={config['kv_cache']}, dtype={config['dtype']}, "
              f"temp={config['temperature']}, max_tokens={config['max_tokens']}")
        
        for img_idx, item in enumerate(BENCHMARK):
            print(f"    Image {img_idx+1}/{len(BENCHMARK)}...", end=" ")
            
            try:
                output, latency, num_tokens, memory = run_inference(
                    model, processor, 
                    item["image_path"], 
                    item["prompt"], 
                    config
                )
                
                result = {
                    "config_name": config["name"],
                    "kv_cache": config["kv_cache"],
                    "dtype": str(config["dtype"]),
                    "temperature": config["temperature"],
                    "max_tokens": config["max_tokens"],
                    "image_idx": img_idx,
                    "prompt": item["prompt"],
                    "output": output,
                    "expected": item["expected"],
                    "latency_ms": latency,
                    "tokens_generated": num_tokens,
                    "latency_per_token_ms": latency / num_tokens if num_tokens > 0 else 0,
                    "memory_mb": memory,
                }
                results.append(result)
                
                print(f"✓ {latency:.0f}ms ({num_tokens} tokens)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        print()
    
    # Save results
    print("Step 4: Saving results...")
    
    # Save raw JSON
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved to {OUTPUT_DIR}/results.json")
    
    # Create summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    summary = {}
    for config in CONFIGS:
        config_results = [r for r in results if r["config_name"] == config["name"]]
        if config_results:
            avg_latency = np.mean([r["latency_ms"] for r in config_results])
            avg_tokens = np.mean([r["tokens_generated"] for r in config_results])
            avg_latency_per_token = np.mean([r["latency_per_token_ms"] for r in config_results])
            avg_memory = np.mean([r["memory_mb"] for r in config_results])
            
            summary[config["name"]] = {
                "avg_latency_ms": avg_latency,
                "avg_tokens": avg_tokens,
                "avg_latency_per_token_ms": avg_latency_per_token,
                "avg_memory_mb": avg_memory,
            }
            
            print(f"\n{config['name'].upper()}:")
            print(f"  Average latency: {avg_latency:.2f} ms")
            print(f"  Average tokens: {avg_tokens:.1f}")
            print(f"  Latency per token: {avg_latency_per_token:.2f} ms/token")
            print(f"  Memory used: {avg_memory:.2f} MB")
    
    # Save summary
    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("="*80)
    
    # Print key findings
    print("\nKEY FINDINGS:")
    if "baseline" in summary and "no_kv_cache" in summary:
        speedup = summary["no_kv_cache"]["avg_latency_ms"] / summary["baseline"]["avg_latency_ms"]
        print(f"  • KV-cache provides {speedup:.2f}x speedup")
    
    if "baseline" in summary and "fp32" in summary:
        speedup = summary["fp32"]["avg_latency_ms"] / summary["baseline"]["avg_latency_ms"]
        print(f"  • FP16 is {speedup:.2f}x faster than FP32")
    
    print("\nNext steps:")
    print("  1. Review outputs in results.json")
    print("  2. Run visualization: python visualize_results.py")
    print("  3. Use data for your paper!")

if __name__ == "__main__":
    main()