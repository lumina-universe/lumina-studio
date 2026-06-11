import sys
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_local_response(args):
    model_name = args.model
    adapter_path = args.adapter
    prompt = args.prompt
    max_tokens = args.max_tokens or 128
    temp = args.temperature or 0.7
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Using device: {device}", file=sys.stderr)
    
    print(f"Loading tokenizer: {model_name}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading base model: {model_name}...", file=sys.stderr)
    dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None
    )
    
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter: {adapter_path}...", file=sys.stderr)
        model = PeftModel.from_pretrained(model, adapter_path)
        
    if device == "mps" and not hasattr(model, "device_map"):
        model = model.to("mps")
        
    print("Generating completion...", file=sys.stderr)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Run generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            do_sample=True if temp > 0.0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    # Extract only new generated tokens (exclude input prompt)
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(response)

def main():
    parser = argparse.ArgumentParser(description="Local Model Inference Pipeline")
    parser.add_argument("--model", required=True, help="Base model name/path")
    parser.add_argument("--adapter", help="Optional path to PEFT adapter weights")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    
    try:
        generate_local_response(args)
    except Exception as e:
        print(f"FAILURE: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
