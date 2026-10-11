import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    dataset = Path("data/test.jsonl")
    output = Path("results/baseline_v1.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    records = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with output.open("w", encoding="utf-8") as f:
        for index, record in enumerate(records[:args.limit], start=1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record["instruction"]},
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=180,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[-1]

            response = tokenizer.decode(
                generated[0][input_length:],
                skip_special_tokens=True,
            ).strip()

            result = {
                "instruction": record["instruction"],
                "expected": record["output"],
                "predicted": response,
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{index}/{min(args.limit, len(records))}] {response}")

    print(f"\nРезультаты сохранены: {output}")


if __name__ == "__main__":
    main()
