"""Обучение LoRA для нормализации клиентских запросов (Qwen2.5-0.5B)."""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = """Ты преобразуешь русскоязычные запросы клиентов
по компьютерной технике и IT-услугам в JSON.

Верни только JSON с полями:
intent, item_type, item, brand, quantity, location, deadline.

intent: price_request, service_order, availability_request, unknown.
item_type: product, service или null.
Не придумывай отсутствующие сведения. Используй null.
quantity — число или null.
deadline сохраняй в исходной формулировке.
"""


class RequestDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.samples = []
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        for number, record in enumerate(records, start=1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record["instruction"]},
            ]
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            if not isinstance(prompt_ids, list):
                prompt_ids = prompt_ids["input_ids"]
            answer = json.dumps(record["output"], ensure_ascii=False, separators=(",", ":"))
            answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(prompt_ids) + len(answer_ids) > max_length:
                raise ValueError(
                    f"{path}, запись {number}: {len(prompt_ids) + len(answer_ids)} токенов "
                    f"превышает max_length={max_length}; увеличьте лимит, не обрезая ответ"
                )
            self.samples.append({
                "input_ids": prompt_ids + answer_ids,
                "labels": [-100] * len(prompt_ids) + answer_ids,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def make_collate(pad_id):
    def collate(batch):
        length = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for sample in batch:
            padding = length - len(sample["input_ids"])
            input_ids.append(sample["input_ids"] + [pad_id] * padding)
            attention_mask.append([1] * len(sample["input_ids"]) + [0] * padding)
            labels.append(sample["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            total_loss += loss.item() * tokens
            total_tokens += tokens
    model.train()
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--output", default="adapters/normalizer_lora_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = полный запуск; 2 = пробный запуск")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Обучение на CPU для этого запуска отключено.")
    if args.grad_accum < 1 or args.epochs < 1:
        raise ValueError("epochs и grad-accum должны быть положительными")
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = RequestDataset(args.train, tokenizer, args.max_length)
    val_data = RequestDataset(args.validation, tokenizer, args.max_length)
    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=collate)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    model.to(device)
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate
    )
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad(set_to_none=True)
    model.train()
    global_steps = 0
    history = []
    stop = False
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Некорректный loss: {loss.item()}")
            running_loss += loss.item()
            scaler.scale(loss / args.grad_accum).backward()
            if index % args.grad_accum == 0 or index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_steps += 1
                if global_steps % 5 == 0 or args.max_steps:
                    print(f"Эпоха {epoch}, шаг {global_steps}, loss={loss.item():.4f}", flush=True)
                if args.max_steps and global_steps >= args.max_steps:
                    stop = True
                    break
        train_loss = running_loss / index
        val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "steps": global_steps, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"Эпоха {epoch}: train_loss={train_loss:.4f}, validation_loss={val_loss:.4f}", flush=True)
        if stop:
            break

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "training_metrics.json").write_text(
        json.dumps({"model": MODEL_ID, "smoke_test": bool(args.max_steps), "history": history},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Адаптер и метрики сохранены: {output}")


if __name__ == "__main__":
    main()
