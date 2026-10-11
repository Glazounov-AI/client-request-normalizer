"""Проверка JSONL-датасета нормализации клиентских запросов."""

import argparse
import json
from pathlib import Path

FIELDS = {"intent", "item_type", "item", "brand", "quantity", "location", "deadline"}
INTENTS = {"price_request", "service_order", "availability_request", "unknown"}
ITEM_TYPES = {"product", "service"}


def validate(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")

    errors = []
    seen = set()
    count = 0

    with path.open(encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            count += 1
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or set(record) != {"instruction", "output"}:
                    raise ValueError("ожидаются только ключи instruction и output")
                instruction = record["instruction"]
                output = record["output"]
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError("instruction должен быть непустой строкой")
                if instruction.strip().casefold() in seen:
                    raise ValueError("дублирующийся instruction")
                seen.add(instruction.strip().casefold())
                if isinstance(output, str):
                    output = json.loads(output)
                if not isinstance(output, dict) or set(output) != FIELDS:
                    raise ValueError("output должен содержать ровно семь полей схемы")
                if output["intent"] not in INTENTS:
                    raise ValueError("недопустимый intent")
                if output["item_type"] is not None and output["item_type"] not in ITEM_TYPES:
                    raise ValueError("недопустимый item_type")
                if output["quantity"] is not None and (type(output["quantity"]) is not int or output["quantity"] <= 0):
                    raise ValueError("quantity должен быть положительным целым числом или null")
                for field in ("item", "brand", "location", "deadline"):
                    value = output[field]
                    if value is not None and (not isinstance(value, str) or not value.strip()):
                        raise ValueError(f"{field} должен быть непустой строкой или null")
                if output["intent"] == "unknown" and any(output[k] is not None for k in FIELDS - {"intent"}):
                    raise ValueError("для unknown все остальные поля должны быть null")
                if output["intent"] != "unknown" and (output["item_type"] is None or output["item"] is None):
                    raise ValueError("для известного intent обязательны item_type и item")
            except (ValueError, json.JSONDecodeError, TypeError) as exc:
                errors.append(f"Строка {line_no}: {exc}")

    if errors:
        print("\n".join(errors))
        raise SystemExit(f"Проверка не пройдена: {len(errors)} ошибок")
    print(f"Проверка пройдена: {count} записей, ошибок нет")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка JSONL-датасета")
    parser.add_argument("path", type=Path, help="Путь к файлу JSONL")
    validate(parser.parse_args().path)
