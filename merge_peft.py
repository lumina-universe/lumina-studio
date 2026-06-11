import sys
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_adapters(args):
    model_name = args.model
    adapter_path = args.adapter
    output_dir = args.output
    
    if not os.path.exists(adapter_path):
        print(f"FAILURE: Adapter directory not found at {adapter_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading tokenizer: {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"Loading base model: {model_name}...", flush=True)
    # Merging needs CPU loading for reliability and compatibility, since MPS/CUDA weights
    # are copied during merge.
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None # Load on CPU for merging
    )
    
    print(f"Loading adapter weights: {adapter_path}...", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging PEFT adapter layers into base model weights...", flush=True)
    merged_model = model.merge_and_unload()
    
    print(f"Saving fully merged standalone model to {output_dir}...", flush=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("SUCCESS: Standalone merged model saved successfully.", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Merge PEFT Adapter into Base Model")
    parser.add_argument("--model", required=True, help="Base model name/path")
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory path")
    parser.add_argument("--output", required=True, help="Output directory for merged model")
    
    args = parser.parse_args()
    
    try:
        merge_adapters(args)
    except Exception as e:
        print(f"FAILURE: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
