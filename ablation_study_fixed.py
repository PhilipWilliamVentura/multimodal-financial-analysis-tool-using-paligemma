# ablation_study_arxiv_ready.py
# PUBLICATION-READY VERSION - LONG SEQUENCES ONLY

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
OUTPUT_DIR = "ablation_results_arxiv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CRITICAL: Long sequences for steady-state measurement
WARMUP_TOKENS = 32  # Exclude prefill + cache warmup
MAX_TOKENS = 512    # Force long generation for true steady-state

# FOCUSED CONFIGS: Architectural ablations only
CONFIGS = [
    {"name": "baseline_kv_cache", "kv_cache": True, "dtype": torch.float16, "temperature": 0.8, "max_tokens": MAX_TOKENS},
    {"name": "no_kv_cache", "kv_cache": False, "dtype": torch.float16, "temperature": 0.8, "max_tokens": MAX_TOKENS},
]

# Diverse prompts to force long generation
BENCHMARK = [
    {
        "url": "https://images.unsplash.com/photo-1574158622682-e40e69881006",
        "prompt": "Describe this image in detail, including the animal's appearance, surroundings, lighting, and mood",
        "expected": "detailed cat description"
    },
    {
        "url": "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e",
        "prompt": "Write a detailed description of this vehicle, including its color, style, surroundings, and any notable features",
        "expected": "detailed car description"
    },
    {
        "url": "https://images.unsplash.com/photo-1551963831-b3b1ca40c98e",
        "prompt": "Describe everything you see in this image, including the food items, presentation, colors, and setting",
        "expected": "detailed food description"
    },
    {
        "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
        "prompt": "Provide a comprehensive description of this landscape, including terrain, sky, lighting, atmosphere, and visual composition",
        "expected": "detailed landscape description"
    },
    {
        "url": "https://images.unsplash.com/photo-1511367461989-f85a21fda167",
        "prompt": "Analyze this image thoroughly, describing the subject, lighting, composition, mood, and any artistic elements",
        "expected": "detailed portrait analysis"
    },
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

def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_peak_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
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

    if kv_cache is not None and kv_cache.num_items() > 0:
        position_ids = attention_mask.cumsum(-1)[:, -1:]
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
    else:
        seq_len = attention_mask.shape[1]
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.masked_fill((attention_mask == 0), 0)

    return final_embedding, causal_mask, position_ids

def patched_rotary_forward(self, x, position_ids, seq_len=None):
    self.inv_freq = self.inv_freq.to(x.device)
    
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    
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
    reset_peak_memory()
    
    # Track steady-state decoding separately
    torch.cuda.synchronize()
    total_start_time = time.perf_counter()
    decode_start_time = None
    decode_start_step = 0
    
    for step in range(config["max_tokens"]):
        # Start measuring steady-state after warmup
        if step == WARMUP_TOKENS:
            torch.cuda.synchronize()
            decode_start_time = time.perf_counter()
            decode_start_step = step
        
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
            pass
        
        if config["kv_cache"]:
            input_ids = next_token.unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
            )
            pixel_values = None
        else:
            all_generated = torch.cat(generated_tokens, dim=-1).unsqueeze(0)
            input_ids = torch.cat([original_input_ids, all_generated], dim=-1)
            attention_mask = torch.cat(
                [original_attention_mask, torch.ones((1, len(generated_tokens)), device=input_ids.device)], dim=-1
            )
            pixel_values = original_pixel_values.to(config["dtype"])
    
    torch.cuda.synchronize()
    total_end_time = time.perf_counter()
    peak_memory = get_peak_memory()
    
    # Calculate metrics
    total_latency_ms = (total_end_time - total_start_time) * 1000
    num_tokens = len(generated_tokens)
    
    # Steady-state tokens/sec (excludes prefill and warmup)
    if decode_start_time is not None and num_tokens > decode_start_step:
        decode_latency_s = total_end_time - decode_start_time
        decode_tokens = num_tokens - decode_start_step
        steady_state_tps = decode_tokens / decode_latency_s if decode_latency_s > 0 else 0
        steady_state_ms_per_token = (decode_latency_s * 1000) / decode_tokens if decode_tokens > 0 else 0
    else:
        # Fallback if generation too short
        steady_state_tps = num_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else 0
        steady_state_ms_per_token = total_latency_ms / num_tokens if num_tokens > 0 else 0
    
    generated_tokens_tensor = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens_tensor, skip_special_tokens=True)
    
    return {
        "output": decoded,
        "total_latency_ms": total_latency_ms,
        "tokens_generated": num_tokens,
        "warmup_tokens": decode_start_step,
        "steady_state_tokens": num_tokens - decode_start_step,
        "peak_memory_mb": peak_memory,
        "steady_state_tps": steady_state_tps,
        "steady_state_ms_per_token": steady_state_ms_per_token,
        "total_ms_per_token": total_latency_ms / num_tokens if num_tokens > 0 else 0
    }

def reset_model_state(model):
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model.zero_grad(set_to_none=True)
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
    
    print("  Applying patches...")
    model._merge_input_ids_with_image_features = types.MethodType(
        patched_merge_input_ids_with_image_features, model
    )
    
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
    print("PALIGEMMA ABLATION STUDY - PUBLICATION-READY")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Warmup tokens (excluded from steady-state): {WARMUP_TOKENS}")
    print(f"Configs: {len(CONFIGS)} architectural ablations")
    print("="*80 + "\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    
    print("Step 1: Downloading images...")
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
    
    print("Step 3: Warmup run...")
    try:
        warmup_image = Image.open(BENCHMARK[0]["image_path"])
        warmup_inputs = processor(text=["warmup"], images=[warmup_image])
        warmup_inputs = move_inputs_to_device(warmup_inputs, DEVICE)
        with torch.no_grad():
            _ = model(**warmup_inputs, kv_cache=None)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        print("✓\n")
    except Exception as e:
        print(f"⚠ Warmup failed: {e}\n")
    
    print("Step 4: Running experiments...")
    print("⚠️  NOTE: Long generations (512 tokens) will take time\n")
    
    results = []
    
    for config_idx, config in enumerate(CONFIGS):
        print(f"\n{'='*80}")
        print(f"CONFIG [{config_idx+1}/{len(CONFIGS)}]: {config['name']}")
        print(f"{'='*80}")
        
        reset_model_state(model)
        
        for img_idx, item in enumerate(BENCHMARK):
            print(f"\n  Image {img_idx+1}/{len(BENCHMARK)}: {item['prompt'][:60]}...")
            print(f"  Running inference...", end=" ", flush=True)
            
            try:
                metrics = run_inference(
                    model, processor, item["image_path"], item["prompt"], config
                )
                
                result = {
                    "config_name": config["name"],
                    "kv_cache": config["kv_cache"],
                    "temperature": config["temperature"],
                    "image_id": img_idx,
                    "prompt": item["prompt"],
                    **metrics
                }
                results.append(result)
                
                print(f"✓")
                print(f"    Total tokens: {metrics['tokens_generated']}")
                print(f"    Steady-state tokens: {metrics['steady_state_tokens']}")
                print(f"    Total latency: {metrics['total_latency_ms']:.0f} ms")
                print(f"    Steady-state ms/token: {metrics['steady_state_ms_per_token']:.1f}")
                print(f"    Steady-state tokens/sec: {metrics['steady_state_tps']:.2f}")
                print(f"    Peak memory: {metrics['peak_memory_mb']:.1f} MB")
                
            except Exception as e:
                print(f"✗ FAILED")
                print(f"    Error: {str(e)[:100]}")
                continue
    
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Raw results: {OUTPUT_DIR}/results.json")
    
    print(f"\n{'='*80}")
    print("COMPUTING SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    summary = {}
    for config in CONFIGS:
        config_results = [r for r in results if r["config_name"] == config["name"]]
        if config_results:
            summary[config["name"]] = {
                "avg_total_latency_ms": round(np.mean([r["total_latency_ms"] for r in config_results]), 2),
                "avg_tokens_generated": round(np.mean([r["tokens_generated"] for r in config_results]), 2),
                "avg_steady_state_tokens": round(np.mean([r["steady_state_tokens"] for r in config_results]), 2),
                "avg_steady_state_ms_per_token": round(np.mean([r["steady_state_ms_per_token"] for r in config_results]), 2),
                "avg_steady_state_tps": round(np.mean([r["steady_state_tps"] for r in config_results]), 2),
                "avg_peak_memory_mb": round(np.mean([r["peak_memory_mb"] for r in config_results]), 2),
                "std_steady_state_tps": round(np.std([r["steady_state_tps"] for r in config_results]), 2),
                "num_samples": len(config_results)
            }
    
    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary: {OUTPUT_DIR}/summary.json\n")
    
    print("="*80)
    print("📊 FINAL RESULTS (ARXIV-READY)")
    print("="*80)
    print(f"\n{'Configuration':<25} {'ms/token':<12} {'tokens/sec':<15} {'Peak VRAM (MB)':<15}")
    print("-" * 80)
    
    for config_name, data in summary.items():
        print(f"{config_name:<25} {data['avg_steady_state_ms_per_token']:<12.1f} "
              f"{data['avg_steady_state_tps']:<15.2f} {data['avg_peak_memory_mb']:<15.1f}")
    
    print("-" * 80)
    
    # Key findings
    if "baseline_kv_cache" in summary and "no_kv_cache" in summary:
        baseline = summary["baseline_kv_cache"]
        no_cache = summary["no_kv_cache"]
        
        tps_speedup = no_cache["avg_steady_state_ms_per_token"] / baseline["avg_steady_state_ms_per_token"]
        
        print("\n" + "="*80)
        print("🔬 KEY FINDINGS FOR PAPER")
        print("="*80)
        print(f"\nKV Cache Impact (Steady-State Decoding):")
        print(f"  • With KV cache:    {baseline['avg_steady_state_ms_per_token']:.1f} ms/token  ({baseline['avg_steady_state_tps']:.1f} tok/s)")
        print(f"  • Without KV cache: {no_cache['avg_steady_state_ms_per_token']:.1f} ms/token  ({no_cache['avg_steady_state_tps']:.1f} tok/s)")
        print(f"  • Speedup:          {tps_speedup:.2f}x")
        print(f"\nMemory (Peak Allocation):")
        print(f"  • Baseline:         {baseline['avg_peak_memory_mb']:.1f} MB")
        print(f"  • No cache:         {no_cache['avg_peak_memory_mb']:.1f} MB")
        print(f"  • Note: Peak includes model weights (5.4 GB) + vision encoder activations")
        print(f"\nGeneration Statistics:")
        print(f"  • Avg tokens:       {baseline['avg_tokens_generated']:.0f}")
        print(f"  • Steady-state:     {baseline['avg_steady_state_tokens']:.0f} tokens (after {WARMUP_TOKENS} warmup)")
        print(f"  • Samples:          {baseline['num_samples']}")
    
    print("\n" + "="*80)
    print("✅ PUBLICATION-READY RESULTS")
    print("="*80)
    print(f"\nAll files saved to: {OUTPUT_DIR}/")
    print("\nNext steps:")
    print("  1. Review summary.json for exact numbers")
    print("  2. Run visualization script for plots")
    print("  3. Use these metrics in your paper")
    print("\n🎓 Ready for arXiv submission!")

if __name__ == "__main__":
    main()