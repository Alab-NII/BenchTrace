#!/usr/bin/env python3
"""
Agent-R: QLoRA fine-tuning of Qwen3-32B on revision trajectories.

Consumes JSONL output of agentR_build_data.py.
Uses standard transformers Trainer (no trl dependency) with a custom
dataset that tokenizes conversations and applies completion-only loss masking:
only tokens inside assistant turns contribute to the loss.

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=0 python agentR_finetune.py \\
      --data agentR_data/zork1.jsonl agentR_data/detective.jsonl \\
      --base_model Qwen/Qwen3-32B \\
      --output_dir output/agentR_ckpt/all_games

After training, merge LoRA for vLLM serving (see pipeline script or docstring):
  python -c "
  import torch; from peft import PeftModel
  from transformers import AutoModelForCausalLM, AutoTokenizer
  base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-32B', torch_dtype=torch.bfloat16, device_map='auto')
  model = PeftModel.from_pretrained(base, 'output/agentR_ckpt/all_games/adapter')
  model.merge_and_unload().save_pretrained('output/agentR_merged')
  AutoTokenizer.from_pretrained('Qwen/Qwen3-32B').save_pretrained('output/agentR_merged')
  "
"""

import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(paths: list[str]) -> list[dict]:
    data = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


class RevisionDataset(Dataset):
    """
    Tokenizes each revision trajectory and applies completion-only loss masking.
    Loss is computed only on assistant response tokens (after <|im_start|>assistant\\n
    up to and including <|im_end|>). All other tokens get label=-100.
    """

    def __init__(self, data: list[dict], tokenizer, max_length: int = 4096):
        # Qwen3 chat format markers
        response_start_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        end_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[-1]
        n = len(response_start_ids)

        self.examples = []
        n_no_loss = 0

        for item in data:
            text = tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            enc = tokenizer(text, max_length=max_length, truncation=True)
            input_ids = enc["input_ids"]

            # Start with all labels masked
            labels = [-100] * len(input_ids)

            # Unmask tokens inside each assistant turn
            has_loss = False
            i = 0
            while i <= len(input_ids) - n:
                if input_ids[i : i + n] == response_start_ids:
                    start = i + n
                    end = len(input_ids)
                    for j in range(start, len(input_ids)):
                        if input_ids[j] == end_token_id:
                            end = j + 1  # include the <|im_end|> token in loss
                            break
                    for k in range(start, end):
                        labels[k] = input_ids[k]
                    has_loss = True
                    i = end
                else:
                    i += 1

            if not has_loss:
                n_no_loss += 1
                continue  # skip examples with no trainable tokens

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

        if n_no_loss:
            print(f"  Skipped {n_no_loss} examples with no assistant tokens in loss mask")
        print(f"  Dataset: {len(self.examples)} trainable examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class PaddingCollator:
    """Pads sequences to the longest in the batch; padded positions get label=-100."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(torch.cat([
                f["input_ids"],
                f["input_ids"].new_full((pad,), self.pad_token_id),
            ]))
            attention_mask.append(torch.cat([
                f["attention_mask"],
                f["attention_mask"].new_zeros(pad),
            ]))
            labels.append(torch.cat([
                f["labels"],
                f["labels"].new_full((pad,), -100),
            ]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True,
                        help="One or more JSONL training data files")
    parser.add_argument("--base_model", default="Qwen/Qwen3-32B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Max optimizer steps. If > 0, overrides --epochs.")
    parser.add_argument("--n_samples", default=-1, type=int,
                        help="Use only the first N samples (by order in file).")
    parser.add_argument("--save_strategy", default="epoch",
                        help="Checkpoint save strategy: epoch, steps, or no.")
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--grad_accum", default=8, type=int)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} revision trajectories from {len(args.data)} file(s)")
    if args.n_samples > 0:
        data = data[:args.n_samples]
        print(f"  Using first {args.n_samples} samples")
    if not data:
        print("No data found. Exiting.")
        return

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model (QLoRA 4-bit) ───────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = RevisionDataset(data, tokenizer, args.max_length)
    if len(dataset) == 0:
        print("No trainable examples after masking. Exiting.")
        return
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    # ── Training ──────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        save_strategy=args.save_strategy,
        save_total_limit=None,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print(f"\nStarting training: {len(dataset)} examples × {args.epochs} epochs")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nLoRA adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
