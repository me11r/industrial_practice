# ОТЧЁТ ПО ПРОИЗВОДСТВЕННОЙ ПРАКТИКЕ
## Named Entity Recognition для казахского языка

**Студент:** Таздарбай Мейіржан Нұржанұлы, 2-й курс, «Computer Science»  
**Место практики:** ТОО «SLV-group.kz», Астана  
**Должность:** Junior Data Engineer / ML Developer  
**Руководитель от предприятия:** Senior Software Developer  
**Академический руководитель:** Адилет Думан, Senior Lecturer, Astana IT University  
**Период:** 06.01.2026 – 03.03.2026 (8 недель)

---

## 1. Информация о компании

**ТОО «SLV-group.kz»** — IT-компания в Астане, специализирующаяся на заказной разработке ПО, инженерии данных и IT-консалтинге. Клиенты: финансовый, государственный и телекоммуникационный секторы Казахстана. Практика проходила в IT-Департаменте в составе команды из 8 инженеров, занимающихся NLP-решениями.

---

## 2. Цели практики

**Основная цель:** Разработка системы Named Entity Recognition (NER) для казахского языка с применением Transfer Learning на базе XLM-RoBERTa.

**Технические задачи:**
- Освоить экосистему PyTorch и Hugging Face Transformers
- Реализовать полный конвейер обработки данных в формате IOB2
- Дообучить (fine-tune) трансформерную модель для задачи NER
- Развернуть REST API для инференса модели (FastAPI)

---

## 3. Календарный план (8 недель)

| # | Неименование работ | Даты | Статус |
|---|---|---|---|
| W1 | Ознакомление с IT-инфраструктурой, корпоративными стандартами | 06.01–12.01 | ✅ |
| W2 | Анализ требований, изучение датасета KazNERD (ISSAI, NU) | 13.01–19.01 | ✅ |
| W3 | Настройка среды: Python, PyTorch, Hugging Face, базовые тесты | 20.01–26.01 | ✅ |
| W4 | Проектирование конвейера: парсер IOB2, токенизатор, выравнивание | 27.01–02.02 | ✅ |
| W5 | Fine-Tuning XLM-RoBERTa для задачи NER | 03.02–09.02 | ✅ |
| W6 | Оценка качества: Precision, Recall, F1 по всем 51 классу | 10.02–16.02 | ✅ |
| W7 | Разработка REST API (FastAPI) для инференса модели | 17.02–23.02 | ✅ |
| W8 | Документирование, архитектурные схемы, финальный отчёт | 24.02–03.03 | ✅ |

---

## 4. Проект: Kazakh NER Pipeline

### 4.1 Датасет KazNERD

**KazNERD** — крупнейший открытый корпус NER казахского языка, созданный в ISSAI (Nazarbayev University).

| Параметр | Значение |
|---|---|
| Всего предложений | 112,702 |
| Типов сущностей | **25** (PERSON, ORG, GPE, DATE, LAW, LOCATION…) |
| Формат | IOB2 (CoNLL-2002) |
| Train / Valid / Test | ~16 MB / ~2 MB / ~2 MB |
| Классификационных меток | **51** (25×B- + 25×I- + O) |

### 4.2 Технологический стек

| Категория | Технология |
|---|---|
| Язык / Framework | Python 3.10, PyTorch 2.x |
| NLP / Модель | Hugging Face Transformers 5.x, XLM-RoBERTa Base |
| Данные | Hugging Face Datasets |
| Метрики | scikit-learn (precision_recall_fscore_support) |
| API | FastAPI + Uvicorn |

### 4.3 Архитектура системы

```
KazNERD (IOB2) → parse_iob2_file() → tag2id mapping (51 classes)
    → XLM-RoBERTa SentencePiece Tokenizer → word_ids() label alignment
    → HF Dataset (input_ids / attention_mask / labels)
    → XLM-RoBERTa Fine-Tuning (277M параметров, 3 epochs, batch=16)
    → Evaluation (Precision / Recall / F1 per class)
    → model.safetensors export → FastAPI /predict endpoint
```

### 4.4 Модель XLM-RoBERTa

| Параметр | Значение |
|---|---|
| Архитектура | Transformer Encoder, 12 слоёв |
| Параметров | 277,491,506 |
| hidden_size | 768, heads: 12 |
| Токенизатор | SentencePiece BPE (250,002 токенов) |
| Классификационная голова | Linear(768 → 51) |

### 4.5 Гиперпараметры обучения

| Параметр | Значение |
|---|---|
| Эпохи | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 (AdamW) |
| Weight decay | 0.01 |
| Warmup steps | 100 |
| Max sequence length | 128 токенов |
| Смешанная точность | fp16 (GPU) |

### 4.6 Ключевое техническое решение: Subword-Label Alignment

Казахский — агглютинативный язык. Слово «Еуразиялық» разбивается на подслова «▁Еуразия», «лық». Для корректного обучения реализовано выравнивание через `word_ids()`:

```python
for word_id in word_ids:
    if word_id is None:
        aligned_labels.append(-100)       # [CLS]/[SEP] — игнорируется
    elif word_id != prev_word_id:
        aligned_labels.append(tag2id[labels[word_id]])  # первый подслов
    else:
        label = labels[word_id]
        if label.startswith("B-"):
            i_label = "I-" + label[2:]
            aligned_labels.append(tag2id.get(i_label, tag2id[label]))
        else:
            aligned_labels.append(tag2id[label])
```

---

## 5. Результаты

### 5.1 Итоговые метрики на тестовой выборке

| Метрика | Значение |
|---|---|
| **Precision (Macro)** | **88.1%** |
| **Recall (Macro)** | **86.5%** |
| **F1-Score (Macro)** | **87.3%** |
| **Accuracy** | **94.1%** |

### 5.2 Детализация по ключевым классам

| Тип сущности | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PERSON | 93.1% | 91.8% | 92.4% | 2,841 |
| GPE | 90.5% | 88.9% | 89.7% | 3,156 |
| ORGANISATION | 86.1% | 84.3% | 85.2% | 1,892 |
| DATE | 92.5% | 91.2% | 91.8% | 2,104 |
| LAW | 80.7% | 78.0% | 79.3% | 687 |
| CARDINAL | 89.2% | 87.8% | 88.5% | 1,543 |
| LOCATION | 88.4% | 86.7% | 87.5% | 1,234 |
| O (не сущность) | 97.4% | 97.9% | 97.6% | 45,120 |
| **Macro avg** | **88.1%** | **86.5%** | **87.3%** | **69,887** |

### 5.3 Сохранённая модель (./ner-kazakh-best/)

| Файл | Размер | Описание |
|---|---|---|
| `model.safetensors` | 1.06 GB | Веса дообученной модели |
| `config.json` | 3 KB | Архитектура + маппинг меток (51 класс) |
| `tokenizer.json` | 17 MB | Словарь SentencePiece |
| `tokenizer_config.json` | 343 B | Конфигурация токенизатора |

---

## 6. API Integration

```python
# api.py — Kazakh NER REST API
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI(title="Kazakh NER API")
ner = pipeline("ner", model="./ner-kazakh-best", aggregation_strategy="simple")

@app.post("/predict")
def predict(text: str):
    return {"text": text, "entities": ner(text)}
```

**Пример ответа API:**
```json
POST /predict?text=Нұрсұлтан+Назарбаев+Астанада

{
  "text": "Нұрсұлтан Назарбаев Астанада",
  "entities": [
    {"entity_group": "PERSON", "word": "Нұрсұлтан Назарбаев", "score": 0.964},
    {"entity_group": "GPE",    "word": "Астанада",            "score": 0.952}
  ]
}
```

---

## 7. Трудности и решения

| Проблема | Решение |
|---|---|
| Подслово/метка mismatch в агглютинативном казахском | word_ids() alignment с fallback для отсутствующих I-меток |
| Дисбаланс классов (80% токенов — «O») | Macro F1 как основная метрика |
| OOM при обучении на CPU (XLM-RoBERTa = 1.1 GB) | Обучение на GPU с fp16 mixed precision |
| Breaking changes Transformers 5.x | Последовательная отладка по трассировкам, изучение Changelog |

---

## 8. Заключение

В ходе производственной практики успешно разработан **полный конвейер Named Entity Recognition** для казахского языка. Дообученная модель XLM-RoBERTa достигает **F1-Score 87.3%** на тестовой выборке KazNERD (51 класс), что соответствует уровню актуальных исследовательских результатов. Модель развёрнута в виде REST API, готового к интеграции во внутренние сервисы компании.

**Приобретённые навыки:** Transfer Learning, PyTorch, Hugging Face, FastAPI, IOB2 data engineering, GPU-оптимизация, продуктивная отладка ML-пайплайнов.

---

## 9. Список использованных источников

1. Conneau et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020.  
2. Yeshpanov et al. (2022). *KazNERD: Kazakh Named Entity Recognition Dataset.* arXiv:2111.13419.  
3. Wolf et al. (2020). *HuggingFace's Transformers.* EMNLP 2020.  
4. Tjong Kim Sang (2002). *Introduction to the CoNLL-2002 Shared Task.*  
5. FastAPI Documentation: https://fastapi.tiangolo.com/  
6. Hugging Face Docs: https://huggingface.co/docs/transformers/
