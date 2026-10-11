"""Оценка baseline и LoRA на эталонном наборе JSONL."""
import argparse
import json
from pathlib import Path

FIELDS = ('intent', 'item_type', 'item', 'brand', 'quantity', 'location', 'deadline')
INTENTS = {'price_request', 'service_order', 'availability_request', 'unknown'}
ITEM_TYPES = {'product', 'service', None}


def parse_object(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            obj = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
        return obj if isinstance(obj, dict) else None
    return None


def valid_schema(obj):
    if obj is None or set(obj) != set(FIELDS):
        return False
    if obj['intent'] not in INTENTS or obj['item_type'] not in ITEM_TYPES:
        return False
    if obj['item'] is not None and not isinstance(obj['item'], str):
        return False
    for key in ('brand', 'location', 'deadline'):
        if obj[key] is not None and not isinstance(obj[key], str):
            return False
    q = obj['quantity']
    return q is None or (isinstance(q, int) and not isinstance(q, bool) and q > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results/baseline_v1.jsonl')
    parser.add_argument('--output', default='results/baseline_metrics_v1.json')
    args = parser.parse_args()

    path = Path(args.input)
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:
        raise ValueError('Файл результатов пуст')

    stats = {'total': len(records), 'json_valid': 0, 'schema_valid': 0,
             'intent_correct': 0, 'fields_correct': 0, 'fields_total': 0,
             'exact_match': 0}
    details = []
    for index, row in enumerate(records, 1):
        expected = parse_object(row['expected'])
        if not valid_schema(expected):
            raise ValueError(f'Некорректный эталон в записи {index}')
        predicted = parse_object(row['predicted'])
        json_ok = predicted is not None
        schema_ok = valid_schema(predicted)
        stats['json_valid'] += json_ok
        stats['schema_valid'] += schema_ok
        stats['fields_total'] += len(FIELDS)
        matched = [key for key in FIELDS if predicted is not None and predicted.get(key, object()) == expected[key]]
        stats['fields_correct'] += len(matched)
        stats['intent_correct'] += 'intent' in matched
        stats['exact_match'] += schema_ok and predicted == expected
        details.append({'index': index, 'instruction': row['instruction'],
                        'json_valid': json_ok, 'schema_valid': schema_ok,
                        'matched_fields': matched, 'field_score': len(matched),
                        'exact_match': bool(schema_ok and predicted == expected)})

    n = stats['total']
    metrics = {
        **stats,
        'json_validity': round(stats['json_valid'] / n, 4),
        'schema_validity': round(stats['schema_valid'] / n, 4),
        'intent_accuracy': round(stats['intent_correct'] / n, 4),
        'field_accuracy': round(stats['fields_correct'] / stats['fields_total'], 4),
        'exact_match_rate': round(stats['exact_match'] / n, 4),
        'details': details,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for label, key in [('JSON Validity', 'json_validity'), ('Schema Validity', 'schema_validity'),
                       ('Intent Accuracy', 'intent_accuracy'), ('Field Accuracy', 'field_accuracy'),
                       ('Exact Match', 'exact_match_rate')]:
        print(f'{label}: {metrics[key]:.1%}')
    print(f'Всего: {n}; метрики: {output}')


if __name__ == '__main__':
    main()
