#!/usr/bin/env python3
"""
SFT LoRA Training Script for Gemma 3 4B (Experiment 4).

Fine-tunes google/gemma-3-4b-it on expected tool-use JSON trajectories using TRL (SFTTrainer) + PEFT.
Saves fine-tuned LoRA adapter to GCS mount directory (/mnt/gcs/adapters/gemma-3-4b-tau-sft).
"""

import os
import sys

def main():
    print("==================================================")
    print("Starting Gemma 3 4B LoRA Fine-Tuning Pipeline (SFT)")
    print("==================================================")

    model_name = os.environ.get("MODEL_NAME", "google/gemma-3-4b-it")
    dataset_path = os.environ.get("DATASET_PATH", "benchmarks/capacity/datasets/sft_tau_dataset.jsonl")
    output_adapter_dir = os.environ.get("OUTPUT_ADAPTER_DIR", "/mnt/gcs/adapters/gemma-3-4b-tau-sft")

    print(f"Base Model    : {model_name}")
    print(f"Dataset Path  : {dataset_path}")
    print(f"Output Adapter: {output_adapter_dir}")

    # Imports wrapped inside main to prevent import errors when running outside training env
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer
    except ImportError as e:
        print(f"Training dependencies not installed: {e}")
        print("Required: torch transformers datasets peft trl")
        sys.exit(0)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    training_args = TrainingArguments(
        output_dir="/tmp/sft_output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        bf16=True,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="messages",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()

    os.makedirs(output_adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(output_adapter_dir)
    tokenizer.save_pretrained(output_adapter_dir)
    print(f"LoRA adapter successfully saved to {output_adapter_dir}")


if __name__ == "__main__":
    main()
