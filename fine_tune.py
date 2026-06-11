import sys
import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset

class JSONProgressCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Avoid sending empty logs or logs without training loss
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            epoch = state.epoch if state.epoch is not None else 0.0
            
            progress = {
                "step": state.global_step,
                "max_steps": state.max_steps,
                "epoch": round(epoch, 4),
                "loss": loss,
                "learning_rate": lr,
                "percent_complete": round((state.global_step / state.max_steps) * 100.0, 2) if state.max_steps > 0 else 0
            }
            # Print with a distinct prefix for easy parser matching
            print(f"METRIC_LOG:{json.dumps(progress)}", flush=True)

def train_model(config_path):
    print(f"Loading configuration from {config_path}...", flush=True)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    model_name = config.get("model_name", "facebook/opt-125m")
    dataset_path = config.get("dataset_path")
    output_dir = config.get("output_dir", "./model_output")
    
    # LoRA Hyperparameters
    lora_r = config.get("lora_r", 8)
    lora_alpha = config.get("lora_alpha", 16)
    lora_dropout = config.get("lora_dropout", 0.05)
    target_modules = config.get("target_modules", ["q_proj", "v_proj"])
    
    # Training Hyperparameters
    epochs = config.get("epochs", 1)
    batch_size = config.get("batch_size", 1)
    learning_rate = config.get("learning_rate", 2e-4)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
    max_seq_length = config.get("max_seq_length", 512)
    max_steps = config.get("max_steps", -1) # -1 means train for all epochs
    
    print(f"Checking hardware accelerator...", flush=True)
    # Detect hardware
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA GPU detected. Training on CUDA.", flush=True)
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Apple Silicon GPU (MPS) detected. Training on MPS.", flush=True)
    else:
        print("No GPU detected. Training on CPU (this might be slow).", flush=True)
    
    # Hugging Face token
    hf_token = config.get("hf_token")
    if hf_token and hf_token.strip():
        os.environ["HF_TOKEN"] = hf_token
        hf_token = hf_token.strip()
    else:
        hf_token = None
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
        
    print(f"Loading tokenizer: {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Left padding is usually preferred for generation, right for training SFT
    tokenizer.padding_side = "right"
    
    print(f"Loading dataset: {dataset_path}...", flush=True)
    if not dataset_path or not os.path.exists(dataset_path):
        # Create a toy dataset if none provided
        print("Dataset file not found or not provided. Generating a small synthetic dataset for training...", flush=True)
        synthetic_data = [
            {"text": "### User: Hello! Who are you?\n### Assistant: I am a fine-tuned local assistant model built to help you."},
            {"text": "### User: What is Hugging Face?\n### Assistant: Hugging Face is a platform that provides open-source tools for machine learning and AI."},
            {"text": "### User: Tell me about OpenRouter.\n### Assistant: OpenRouter is a unified API service that provides access to many LLMs with a single endpoint."},
            {"text": "### User: What is fine-tuning?\n### Assistant: Fine-tuning is the process of training a pre-trained model on a specific dataset to adapt it to a specific task."},
            {"text": "### User: Explain LoRA.\n### Assistant: LoRA stands for Low-Rank Adaptation. It is a parameter-efficient fine-tuning method that adds small trainable matrices to the model."}
        ] * 4 # Duplicate to make it run for a few steps
        dataset = Dataset.from_list(synthetic_data)
    else:
        # Load from file (supports JSON, JSONL, or CSV)
        if dataset_path.endswith(".csv"):
            dataset = load_dataset("csv", data_files=dataset_path, split="train")
        else:
            dataset = load_dataset("json", data_files=dataset_path, split="train")
            
    print(f"Dataset loaded. Total rows: {len(dataset)}", flush=True)
    
    print(f"Loading base model: {model_name}...", flush=True)
    # Map model to device. On MPS, AutoModelForCausalLM usually needs to be loaded to cpu first then moved,
    # or loaded directly with low_cpu_mem_usage.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.float32, # CPU and MPS are reliable with FP32
        device_map=device if device != "cpu" else None
    )
    
    print(f"Configuring LoRA parameters (Rank={lora_r}, Alpha={lora_alpha})...", flush=True)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    print(f"Setting up SFTConfig parameters...", flush=True)
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=1,
        num_train_epochs=epochs,
        max_steps=max_steps,
        fp16=False,
        bf16=False,
        save_strategy="no",
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        remove_unused_columns=True,
        dataset_text_field="text",
        max_length=max_seq_length
    )
    
    # SFT Trainer
    print(f"Initializing SFTTrainer...", flush=True)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[JSONProgressCallback()]
    )
    
    print("Starting training loop...", flush=True)
    trainer.train()
    
    print(f"Training complete! Saving fine-tuned adapter to {output_dir}...", flush=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("SUCCESS: Model adapter and tokenizer saved successfully.", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Local LoRA Fine-Tuning CLI")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file")
    args = parser.parse_args()
    
    try:
        train_model(args.config)
    except Exception as e:
        print(f"FAILURE: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
