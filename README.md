# Client Request Normalizer

**Нормализация русскоязычных клиентских запросов в структурированный JSON с помощью LoRA.**

Проект исследует, насколько дообучение компактной языковой модели улучшает извлечение информации из неформальных обращений клиентов в области компьютерной техники и IT-услуг.

## О проекте

Клиенты пишут свободным текстом: спрашивают стоимость, заказывают услуги, уточняют наличие товаров или задают вопросы, которые не относятся к этим категориям. Модель преобразует сообщение в единый JSON-формат, пригодный для последующей обработки в CRM, чат-боте или системе заявок.

**Пример входа:**

> Сколько стоит установить Windows на два ноутбука?

**Ожидаемый выход:**

```json
{
  "intent": "price_request",
  "item_type": "service",
  "item": "установка Windows",
  "brand": null,
  "quantity": 2,
  "location": null,
  "deadline": null
}
```

Модель **не рассчитывает цены и не проверяет остатки**: она выделяет данные для дальнейшей бизнес-логики. Неуказанные сведения должны оставаться `null`.

## Возможности

- Классификация намерения: `price_request`, `service_order`, `availability_request`, `unknown`.
- Определение типа позиции (`product` или `service`) и извлечение наименования, бренда, количества, места и срока.
- Формирование ответа в виде JSON с семью фиксированными полями.
- Сравнение исходной модели и версии с LoRA на одном тестовом наборе.
- Автоматическая проверка синтаксиса JSON, соответствия схеме и точности извлечения.

Текущая схема рассчитана на **одну основную позицию** в запросе. Обращения с несколькими разными товарами требуют отдельного решения в будущей версии.

## Технологии

- Python 3.12, PyTorch с CUDA.
- `Qwen/Qwen2.5-0.5B-Instruct` — базовая модель.
- Hugging Face Transformers, PEFT (LoRA), Datasets, Accelerate.
- Локальное обучение и тестирование на NVIDIA GeForce RTX 3050 (8 ГБ VRAM).
- Git — фиксация датасета, кода и результатов эксперимента.

## Подготовка и обучение

Из корня проекта активируйте виртуальное окружение и установите зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

> Для работы на GPU требуется совместимая сборка PyTorch с поддержкой CUDA. Состав зависимостей зафиксирован в `requirements.txt`; доступность GPU можно проверить скриптом `src/check_model.py`.

Проверка данных:

```powershell
python src/validate_dataset.py data/train.jsonl
python src/validate_dataset.py data/validation.jsonl
python src/validate_dataset.py data/test.jsonl
```

Пробное обучение и полноценный запуск LoRA:

```powershell
python src/train_lora.py --max-steps 2 --output adapters/smoke_test
python src/train_lora.py --output adapters/lora_v1
```

Для v1 использовались три эпохи (69 шагов оптимизации), LoRA rank 8, alpha 16, FP16 и накопление градиентов. Адаптеры сохраняются локально в `adapters/` и не включаются в Git.

## Тестирование и результаты

Тестовый набор содержит **15 запросов**, не использованных для обучения. При сравнении применялись одинаковые тестовые запросы и параметры генерации (`do_sample=False`).

Получение ответов исходной модели и LoRA:

```powershell
python src/baseline.py --limit 15 --output results/baseline_v1.jsonl
python src/baseline.py --limit 15 --adapter adapters/lora_v1 --output results/lora_v1.jsonl
```

Расчёт метрик:

```powershell
python src/evaluate.py --input results/baseline_v1.jsonl --output results/baseline_metrics_v1.json
python src/evaluate.py --input results/lora_v1.jsonl --output results/lora_metrics_v1.json
```

| Метрика | Baseline | LoRA v1 |
|---|---:|---:|
| JSON Validity | 93,3% | **100,0%** |
| Schema Validity | 66,7% | **73,3%** |
| Intent Accuracy | 53,3% | **93,3%** |
| Field Accuracy | 39,1% | **85,7%** |
| Exact Match | 0,0% | **46,7%** |

На 15 тестовых запросах LoRA правильно определила намерение в 14 случаях и полностью совпала с эталоном в 7 случаях. Результаты и детализация сохранены в `results/`.

**Ограничения эксперимента:** выборка мала и синтетическая; она не доказывает устойчивость на реальных обращениях. В отдельных ответах сохраняются нарушения схемы (например, объект вместо строки в `location`, отсутствие обязательного поля), ошибки нормализации наименований и сложности с несколькими позициями. Падение обучающей ошибки само по себе не гарантирует качества на новых данных. Для дальнейшего развития нужны более разнообразные примеры, расширенные тесты и проверка структурированного вывода.

## Бизнес-ценность

Нормализованный JSON может служить промежуточным слоем между свободным текстом клиента и прикладной системой: CRM, каталогом товаров, обработчиком заявок или чат-ботом. Это помогает отделить распознавание намерения от поиска цен, наличия и исполнения заказа.

**Статус:** завершённый эксперимент LoRA v1 и прототип компонента нормализации; не готовая производственная система.

---

## English

### Overview

**Client Request Normalizer** is an experimental Russian-language request-to-JSON extraction project for computer hardware and IT services. It compares the original `Qwen/Qwen2.5-0.5B-Instruct` model with a locally trained LoRA adapter.

### Features

Intent classification (`price_request`, `service_order`, `availability_request`, `unknown`), product/service identification, and extraction of item, brand, quantity, location, and deadline into a fixed seven-field JSON schema. Missing information must remain `null`.

### Technology Stack

Python, PyTorch/CUDA, Transformers, PEFT (LoRA), Datasets, Accelerate; trained locally on an NVIDIA RTX 3050 8 GB GPU.

### Training and Evaluation

The synthetic dataset contains 120 examples: 90 training, 15 validation, and 15 test records. On the 15-example test set, field accuracy increased from **39.1% to 85.7%**, intent accuracy from **53.3% to 93.3%**, and exact match from **0% to 46.7%**. These are small-sample experimental results, not production benchmarks.

### Business Value

The component can support downstream CRM, helpdesk, or chatbot workflows by transforming informal requests into structured records. Price lookup, inventory checks, and transaction execution are outside the scope of this prototype.
