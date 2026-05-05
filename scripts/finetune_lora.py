"""
Fine-tune LLaMA 3 8B for resume-JD relevance scoring using LoRA.

Training task: binary classification (Yes/No) given a (JD, resume) pair.

Data format expected in --data_path (JSONL):
  {"job_description": "...", "resume": "...", "label": 1}   ← relevant
  {"job_description": "...", "resume": "...", "label": 0}   ← not relevant

Usage
-----
  python scripts/finetune_lora.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --data_path data/train.jsonl \
    --output_dir models/lora_reranker \
    --epochs 3 \
    --batch_size 4 \
    --lr 2e-4

Requirements
------------
  pip install transformers peft bitsandbytes accelerate datasets trl
"""
import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are an expert technical recruiter. Evaluate resume-job fit.\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "Job Description:\n{jd}\n\n"
    "Resume:\n{resume}\n\n"
    "Is this resume a strong match for the job? Answer with only 'Yes' or 'No'.\n"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    "{answer}<|eot_id|>"
)


def load_dataset_from_jsonl(path: str, tokenizer, max_length: int = 512):
    from datasets import Dataset

    records = []
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            answer = "Yes" if item["label"] == 1 else "No"
            text = PROMPT_TEMPLATE.format(
                jd=item["job_description"][:1000],
                resume=item["resume"][:1000],
                answer=answer,
            )
            records.append({"text": text})

    dataset = Dataset.from_list(records)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_path",   required=True)
    parser.add_argument("--output_dir",  default="models/lora_reranker")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--max_length",  type=int,   default=512)
    parser.add_argument("--lora_r",      type=int,   default=16)
    parser.add_argument("--lora_alpha",  type=int,   default=32)
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        BitsAndBytesConfig, TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from trl import SFTTrainer

    # ── load model 4-bit ──────────────────────────────────────────────────
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ───────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── dataset ───────────────────────────────────────────────────────────
    logger.info("Loading dataset from %s", args.data_path)
    train_dataset = load_dataset_from_jsonl(args.data_path, tokenizer, args.max_length)
    logger.info("Training samples: %d", len(train_dataset))

    # ── training ──────────────────────────────────────────────────────────
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=train_args,
        max_seq_length=args.max_length,
        dataset_text_field=None,   # already tokenized
    )

    logger.info("Starting training …")
    trainer.train()

    # ── save ──────────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("LoRA weights saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
