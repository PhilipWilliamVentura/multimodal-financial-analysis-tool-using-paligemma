# ablation_study_fixed.py
# WORKING VERSION - MONKEY PATCHES APPLIED CORRECTLY

import torch
import time
import json
import os
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import types

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer

MODEL_PATH = r"C:\Users\Philip Ventura\projects\paligemma-weights\paligemma-3b-pt-224"
OUTPUT_DIR = "ablation_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIGS = [
    {"name": "baseline", "kv_cache": True, "dtype": torch.float16, "temperature": 0.8, "max_tokens": 100},
    {"name": "no_kv_cache", "kv_cache": False, "dtype": torch.float16, "temperature": 0.8, "max_tokens": 100},
    {"name": "temp_0", "kv_cache": True, "dtype": torch.float16, "temperature": 0.0, "max_tokens": 100},
    {"name": "temp_1", "kv_cache": True, "dtype": torch.float16, "temperature": 1.0, "max_tokens": 100},
    {"name": "short_gen", "kv_cache": True, "dtype": torch.float16, "temperature": 0.8, "max_tokens": 50},
    {"name": "long_gen", "kv_cache": True, "dtype": torch.float16, "temperature": 0.8, "max_tokens": 200},
]

BENCHMARK = [
    {"url": "https://images.unsplash.com/photo-1574158622682-e40e69881006", "prompt": "What animal is this?", "expected": "cat"},
    {"url": "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e", "prompt": "What color is the car?", "expected": "red"},
    {"url": "https://images.unsplash.com/photo-1551963831-b3b1ca40c98e", "prompt": "What food is shown?", "expected": "breakfast"},
    {"url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4", "prompt": "Describe the landscape", "expected": "mountains"},
    {"url": "https://images.unsplash.com/photo-1511367461989-f85a21fda167", "prompt": "What is the main subject?", "expected": "profile"},
]

def download_image(url, save_path):
    try:
        response = requests.get(url + "?w=400", timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img.save(save_path)
        return img
    except Exception as e:
        print(f"Error: {e}")
        img = Image.new('RGB', (224, 224), color='red')
        img.save(save_path)
        return img

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def move_inputs_to_device(model_inputs, device):
    return {k: v.to(device) for k, v in model_inputs.items()}

def _sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

# PATCH FUNCTIONS THAT WILL REPLACE THE ORIGINALS
def patched_merge_input_ids_with_image_features(
    self,
    image_features,
    inputs_embeds,
    input_ids,
    attention_mask,
    kv_cache=None,
):
    _, _, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape
    dtype, device = inputs_embeds.dtype, inputs_embeds.device

    scaled_image_features = image_features / (self.config.hidden_size ** 0.5)
    final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=dtype, device=device)

    text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
    image_mask = input_ids == self.config.image_token_index
    pad_mask = input_ids == self.pad_token_id

    text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
    pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
    image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

    final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
    final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
    final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

    q_len = inputs_embeds.shape[1]
    if kv_cache is None or kv_cache.num_items() == 0:
        causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
    else:
        assert q_len == 1
        kv_len = kv_cache.num_items() + q_len
        causal_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)

    causal_mask = causal_mask.unsqueeze(1)

    # FIXED: Single, correct position_ids calculation
    if kv_cache is not None and kv_cache.num_items() > 0:
        # With cache: position = cumsum of attention up to current token
        position_ids = attention_mask.cumsum(-1)[:, -1:]
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
    else:
        # Without cache: sequential 0,1,2,... for all tokens
        seq_len = attention_mask.shape[1]
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        # Zero out padding positions
        position_ids = position_ids.masked_fill((attention_mask == 0), 0)

    return final_embedding, causal_mask, position_ids

def patched_rotary_forward(self, x, position_ids, seq_len=None):
    self.inv_freq = self.inv_freq.to(x.device)
    
    # Ensure 2D
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    
    # Clamp to valid range
    max_pos = self.max_position_embeddings - 1
    position_ids = torch.clamp(position_ids, 0, max_pos)
    
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def run_inference(model, processor, image_path, prompt, config):
    image = Image.open(image_path)
    model_inputs = processor(text=[prompt], images=[image])
    model_inputs = move_inputs_to_device(model_inputs, DEVICE)
    
    # CRITICAL: Save original inputs for no_kv_cache case
    original_input_ids = model_inputs["input_ids"].clone()
    original_attention_mask = model_inputs["attention_mask"].clone()
    original_pixel_values = model_inputs["pixel_values"].clone()
    
    input_ids = original_input_ids
    attention_mask = original_attention_mask
    pixel_values = original_pixel_values
    
    model = model.to(config["dtype"])
    pixel_values = pixel_values.to(config["dtype"])
    
    kv_cache = KVCache() if config["kv_cache"] else None
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mem_before = get_gpu_memory()
    start_time = time.time()
    
    for step in range(config["max_tokens"]):
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
        
        # CRITICAL FIX: Different handling for KV cache vs no cache
        if config["kv_cache"]:
            # With cache: only pass the new token
            input_ids = next_token.unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
            )
            pixel_values = None  # Don't pass image again
        else:
            # Without cache: rebuild FULL sequence INCLUDING new token
            # Concatenate all generated tokens to original input
            all_generated = torch.cat(generated_tokens, dim=-1).unsqueeze(0)
            input_ids = torch.cat([original_input_ids, all_generated], dim=-1)
            attention_mask = torch.cat(
                [original_attention_mask, torch.ones((1, len(generated_tokens)), device=input_ids.device)], dim=-1
            )
            # Pass image on EVERY forward pass (no cache means recompute vision)
            pixel_values = original_pixel_values.to(config["dtype"])
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    mem_after = get_gpu_memory()
    memory_used = mem_after - mem_before
    
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return decoded, latency_ms, len(generated_tokens), memory_used

def reset_model_state(model):
    """Clear any cached state to prevent contamination between runs"""
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model.zero_grad(set_to_none=True)
    # Force garbage collection
    import gc
    gc.collect()

def load_model_simple(model_path, device):
    print("  Loading config...")
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = PaliGemmaConfig(**json.load(f))
    
    print("  Creating model...")
    model = PaliGemmaForConditionalGeneration(config)
    
    print("  Loading weights...")
    try:
        import safetensors.torch
        weight_files = sorted(Path(model_path).glob("*.safetensors"))
        
        if not weight_files:
            weight_files = sorted(Path(model_path).glob("*.bin"))
        
        print(f"  Found {len(weight_files)} weight files")
        
        for i, wf in enumerate(weight_files, 1):
            print(f"    [{i}/{len(weight_files)}] {wf.name}...", end="", flush=True)
            if str(wf).endswith(".safetensors"):
                weights = safetensors.torch.load_file(str(wf))
            else:
                weights = torch.load(str(wf), map_location="cpu")
            
            weights = {k: v.half() if v.dtype == torch.float32 else v for k, v in weights.items()}
            model.load_state_dict(weights, strict=False)
            del weights
            print(" ✓")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise
    
    print(f"  Moving to {device}...")
    model = model.to(device=device, dtype=torch.float16)
    model.eval()
    model.tie_weights()
    
    # APPLY PATCHES - Replace methods directly
    print("  Applying patches for no_kv_cache...")
    model._merge_input_ids_with_image_features = types.MethodType(
        patched_merge_input_ids_with_image_features, model
    )
    
    # Patch all rotary embeddings
    for layer in model.language_model.model.layers:
        layer.self_attn.rotary_emb.forward = types.MethodType(
            patched_rotary_forward, layer.self_attn.rotary_emb
        )
    print("  ✓ Patches applied")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return model, tokenizer

def main():
    print("="*80)
    print("PALIGEMMA ABLATION STUDY - PATCHED VERSION")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    
    print("Step 1: Images...")
    for i, item in enumerate(BENCHMARK):
        save_path = f"{OUTPUT_DIR}/images/img_{i}.jpg"
        if not os.path.exists(save_path):
            download_image(item["url"], save_path)
        item["image_path"] = save_path
    print("✓\n")
    
    print("Step 2: Loading model...")
    model, tokenizer = load_model_simple(MODEL_PATH, DEVICE)
    processor = PaliGemmaProcessor(tokenizer, model.config.vision_config.num_image_tokens, model.config.vision_config.image_size)
    print("✓\n")
    
    print("Step 2.5: Warmup...")
    try:
        warmup_image = Image.open(BENCHMARK[0]["image_path"])
        warmup_inputs = processor(text=["warmup"], images=[warmup_image])
        warmup_inputs = move_inputs_to_device(warmup_inputs, DEVICE)
        with torch.no_grad():
            _ = model(**warmup_inputs, kv_cache=None)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        print("✓\n")
    except Exception as e:
        print(f"⚠ {e}\n")
    
    print("Step 3: Experiments...")
    results = []
    
    for config_idx, config in enumerate(CONFIGS):
        print(f"[{config_idx+1}/{len(CONFIGS)}] {config['name']}")
        
        reset_model_state(model)
        
        for img_idx, item in enumerate(BENCHMARK):
            print(f"    {img_idx+1}/5...", end=" ", flush=True)
            
            try:
                output, latency, num_tokens, memory = run_inference(
                    model, processor, item["image_path"], item["prompt"], config
                )
                
                results.append({
                    "config_name": config["name"],
                    "kv_cache": config["kv_cache"],
                    "temperature": config["temperature"],
                    "prompt": item["prompt"],
                    "output": output,
                    "latency_ms": latency,
                    "tokens_generated": num_tokens,
                    "memory_mb": memory,
                })
                print(f"✓ {latency:.0f}ms")
                
            except Exception as e:
                print(f"✗ {str(e)[:50]}")
                continue
        print()
    
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    summary = {}
    for config in CONFIGS:
        config_results = [r for r in results if r["config_name"] == config["name"]]
        if config_results:
            avg_lat = np.mean([r["latency_ms"] for r in config_results])
            summary[config["name"]] = avg_lat
            print(f"{config['name']:15s}: {avg_lat:6.0f} ms")
    
    if "baseline" in summary and "no_kv_cache" in summary:
        speedup = summary["no_kv_cache"] / summary["baseline"]
        print(f"\n⭐ KV cache: {speedup:.1f}x speedup")
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    summary = {}
    for config in CONFIGS:
        config_results = [r for r in results if r["config_name"] == config["name"]]
        if config_results:
            avg_lat = np.mean([r["latency_ms"] for r in config_results])
            avg_tokens = np.mean([r["tokens_generated"] for r in config_results])
            avg_memory = np.mean([r["memory_mb"] for r in config_results])
            
            summary[config["name"]] = {
                "avg_latency_ms": round(avg_lat, 2),
                "avg_tokens": round(avg_tokens, 2),
                "avg_memory_mb": round(avg_memory, 2),
                "num_samples": len(config_results)
            }
            
            print(f"{config['name']:15s}: {avg_lat:6.0f} ms | {avg_tokens:4.1f} tokens | {avg_memory:6.1f} MB")
    
    # Save summary to JSON
    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Calculate speedups
    if "baseline" in summary and "no_kv_cache" in summary:
        speedup = summary["no_kv_cache"]["avg_latency_ms"] / summary["baseline"]["avg_latency_ms"]
        print(f"\n⭐ KV cache speedup: {speedup:.2f}x")
        
        memory_overhead = summary["baseline"]["avg_memory_mb"] / summary["no_kv_cache"]["avg_memory_mb"]
        print(f"⭐ Memory efficiency: {memory_overhead:.2%} (cache uses less incremental memory)")
    
    print(f"\n✓ Results saved to: {OUTPUT_DIR}/results.json")
    print(f"✓ Summary saved to: {OUTPUT_DIR}/summary.json")
    
    print("\nDone!")

if __name__ == "__main__":
    main()