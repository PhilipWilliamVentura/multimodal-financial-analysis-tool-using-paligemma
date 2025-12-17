from transformers import AutoTokenizer
from modeling_gemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration
import torch
import json

def load_hf_model(model_path, device="cuda"):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")

    # Load config
    with open(f"{model_path}/config.json", "r") as f:
        config = PaliGemmaConfig(**json.load(f))

    # **Memory-efficient model loading**
    model = PaliGemmaForConditionalGeneration(config)

    # Use meta tensors + offload layers if needed (huggingface style)
    try:
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from accelerate.utils import get_balanced_memory

        # Initialize empty model on CPU
        with init_empty_weights():
            empty_model = PaliGemmaForConditionalGeneration(config)

        # Automatically calculate memory allocation for CPU/GPU
        max_memory = get_balanced_memory(empty_model, low_zero=True)

        # Load model checkpoint in 8-bit or half precision
        model = load_checkpoint_and_dispatch(
            empty_model,
            model_path,
            device_map="auto",  # splits layers across CPU/GPU
            dtype=torch.float16,  # half precision
            max_memory=max_memory,
            offload_folder="offload", 
            no_split_module_classes=["PaliGemmaBlock"]  # adapt to your module names
        )
    except ImportError:
        print("Accelerate not installed. Loading full model in float16 (may OOM).")
        model.to(device=device, dtype=torch.float16)

    # Tie weights
    model.tie_weights()

    return model, tokenizer
