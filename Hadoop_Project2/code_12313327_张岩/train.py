#!/usr/bin/env python3
"""
Q1(3): Ray Train LoRA Fine-Tuning Script
==========================================
Fine-tune Qwen3.5-0.8B with LoRA on DataMind-12K using Ray Train.
Memory optimized: GPU offloading, gradient checkpointing, mixed precision.
"""

import os
import sys
import json
import time
from contextlib import nullcontext
import torch
import ray
from ray import train
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "qwen_model_local")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "train_data.jsonl")
VAL_DATA_PATH = os.path.join(BASE_DIR, "data", "val_data.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "qwen-datamind-lora-ray")

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

TRAIN_CONFIG = {
    "model_path": MODEL_PATH,
    "train_data_path": TRAIN_DATA_PATH,
    "val_data_path": VAL_DATA_PATH,
    "epochs": 3,
    "batch_size": 1,
    "lr": 5e-5,
    "max_length": 1536,
    "gradient_accumulation_steps": 8,
    "max_grad_norm": 1.0,
}


# ============================================================
# Data Preprocessing
# ============================================================

def preprocess_dataset(dataset, tokenizer, max_length: int = 1536):
    """Tokenize DataMind-12K trajectories with assistant-only loss masking."""

    def tokenize_fn(examples: dict) -> dict:
        input_ids_list = []
        labels_list = []

        asst_start_marker = "<|im_start|>assistant\n"
        asst_end_marker = "<|im_end|>"

        for messages_list in examples["messages"]:
            messages = [{"role": m["role"], "content": m["content"]}
                        for m in messages_list]

            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            encoding = tokenizer(
                full_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            input_ids = encoding["input_ids"]
            offsets = encoding["offset_mapping"]

            labels = [-100] * len(input_ids)

            search_start = 0
            while True:
                marker_start = full_text.find(asst_start_marker, search_start)
                if marker_start == -1:
                    break

                content_start_char = marker_start + len(asst_start_marker)
                content_end_char = full_text.find(asst_end_marker, content_start_char)
                if content_end_char == -1:
                    content_end_char = len(full_text)

                start_token = None
                end_token = None
                for i, (char_s, char_e) in enumerate(offsets):
                    if start_token is None and char_e > content_start_char and char_s >= content_start_char:
                        start_token = i
                    if start_token is not None and end_token is None and char_s >= content_end_char:
                        end_token = i - 1
                        break

                if start_token is None:
                    for i, (char_s, char_e) in enumerate(offsets):
                        if char_s <= content_start_char < char_e:
                            start_token = i
                            break
                if end_token is None:
                    end_token = len(offsets) - 1

                if start_token is not None and end_token is not None and start_token <= end_token:
                    for i in range(start_token, end_token + 1):
                        labels[i] = input_ids[i]

                search_start = content_end_char + len(asst_end_marker)

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "labels": labels_list}

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing trajectories",
    )


# ============================================================
# Training Function (runs on each Ray Train worker)
# ============================================================

def train_func(config: dict):
    model_path = config.get("model_path")
    train_data_path = config.get("train_data_path")
    val_data_path = config.get("val_data_path")
    epochs = config.get("epochs", 3)
    batch_size = config.get("batch_size", 1)
    learning_rate = config.get("lr", 5e-5)
    max_length = config.get("max_length", 1536)
    grad_accum = config.get("gradient_accumulation_steps", 8)
    max_grad_norm = config.get("max_grad_norm", 1.0)

    # Ray Train worker context
    context = train.get_context()
    worker_rank = context.get_world_rank()
    world_size = context.get_world_size()

    print(f"[Worker {worker_rank}/{world_size}] Initializing...", flush=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    dtype = torch.bfloat16 if use_gpu else torch.float32

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if use_gpu:
        model = model.to("cuda")

    # Apply LoRA
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    # Gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if worker_rank == 0:
        print(f"LoRA trainable params: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.2f}%)", flush=True)

    # Load and preprocess datasets
    raw_train = load_dataset("json", data_files=train_data_path, split="train")
    raw_val = load_dataset("json", data_files=val_data_path, split="train")

    if worker_rank == 0:
        print(f"Train samples: {len(raw_train)}, Val samples: {len(raw_val)}", flush=True)

    tokenized_train = preprocess_dataset(raw_train, tokenizer, max_length)
    tokenized_val = preprocess_dataset(raw_val, tokenizer, max_length)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, return_tensors="pt",
    )

    train_loader = DataLoader(
        tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collator,
    )
    val_loader = DataLoader(
        tokenized_val, batch_size=batch_size, shuffle=False, collate_fn=collator,
    )

    # Ray Train DDP wrapping
    train_loader = train.torch.prepare_data_loader(train_loader)
    val_loader = train.torch.prepare_data_loader(val_loader)
    model = train.torch.prepare_model(model)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps_per_epoch = len(tokenized_train) // (batch_size * world_size * grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * 0.03))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_gpu else nullcontext()

    # ============================================================
    # Training Loop
    # ============================================================
    model.train()
    global_step = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            with autocast_ctx:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            loss = outputs.loss / grad_accum
            loss.backward()

            num_batches += 1

            if (num_batches % grad_accum == 0) or (num_batches == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * grad_accum

            if worker_rank == 0 and num_batches % 50 == 0:
                elapsed = time.time() - epoch_start
                current_loss = total_loss / num_batches
                print(f"  [Epoch {epoch+1}] batch {num_batches}/{len(train_loader)} | "
                      f"loss={current_loss:.4f} | elapsed={elapsed:.0f}s", flush=True)

        avg_train_loss = total_loss / max(num_batches, 1)

        # Validation
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_val_loss += outputs.loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / max(num_val_batches, 1)
        model.train()

        current_lr = scheduler.get_last_lr()[0]

        if worker_rank == 0:
            elapsed = time.time() - epoch_start
            print(f"[Epoch {epoch + 1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Step: {global_step} | "
                  f"Time: {elapsed:.0f}s", flush=True)

        # Report metrics to Ray Train (triggers checkpointing)
        train.report({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": current_lr,
            "global_step": global_step,
        })

    print(f"[Worker {worker_rank}] Training complete.", flush=True)


# ============================================================
# Main Launcher
# ============================================================

def main():
    if not ray.is_initialized():
        ray.init()

    dashboard_url = getattr(ray.get_runtime_context(), "dashboard_url", None) or "not enabled"
    print(f"Ray initialized. Dashboard: {dashboard_url}", flush=True)

    use_gpu = torch.cuda.is_available()
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=use_gpu,
    )

    checkpoint_config = CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    storage_path = os.path.join(OUTPUT_PATH, "ray_checkpoints")
    os.makedirs(storage_path, exist_ok=True)

    run_config = RunConfig(
        name="qwen-datamind-lora",
        storage_path=storage_path,
        checkpoint_config=checkpoint_config,
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=TRAIN_CONFIG,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    print("=" * 60, flush=True)
    print("Starting LoRA fine-tuning on DataMind-12K (Ray Train)", flush=True)
    print(f"  Workers: 1, GPU: {use_gpu}", flush=True)
    print(f"  LoRA rank: {LORA_CONFIG['r']}, alpha: {LORA_CONFIG['lora_alpha']}", flush=True)
    print(f"  Epochs: {TRAIN_CONFIG['epochs']}, max_length: {TRAIN_CONFIG['max_length']}", flush=True)
    print(f"  lr: {TRAIN_CONFIG['lr']}, grad_accum: {TRAIN_CONFIG['gradient_accumulation_steps']}", flush=True)
    print(f"  Output: {OUTPUT_PATH}", flush=True)
    print("=" * 60, flush=True)

    result = trainer.fit()

    print("\n" + "=" * 60, flush=True)
    print("Fine-tuning complete!", flush=True)
    print(f"  Experiment path: {result.path}", flush=True)
    print(f"  Final metrics:   {result.metrics}", flush=True)
    print(f"  Checkpoint:      {result.checkpoint}", flush=True)
    print("=" * 60, flush=True)

    ray.shutdown()


if __name__ == "__main__":
    main()
