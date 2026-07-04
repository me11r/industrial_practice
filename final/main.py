"""
Kazakh Named Entity Recognition (NER) Pipeline
================================================
Industrial Practice Project — SLV-group.kz
Student: Tazdarbay Meirzhan, Astana IT University
Period: January 6 – March 3, 2026

End-to-end pipeline:
  1. Parse KazNERD IOB2 dataset
  2. Build tag mappings
  3. Tokenize with XLM-RoBERTa + align labels
  4. Fine-tune transformer for token classification
  5. Evaluate with Precision, Recall, F1-Score
"""

import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, classification_report
import torch
import json

# ============================================================
# STEP 1: Parse IOB2 files (CoNLL-2002 format)
# ============================================================

def parse_iob2_file(file_path, num_sentences=None):
    """
    Parse a CoNLL-2002 IOB2 formatted file into a list of sentences.
    Each sentence is a list of (token, label) tuples.
    """
    sentences = []
    current_sentence = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                    if num_sentences and len(sentences) >= num_sentences:
                        break
                continue
            parts = line.split()
            if len(parts) == 2:
                token, label = parts[0], parts[1]
                current_sentence.append((token, label))

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


print("=" * 60)
print("STEP 1: Loading KazNERD Dataset")
print("=" * 60)

train_sentences = parse_iob2_file("IOB2_train.txt", num_sentences=None)
valid_sentences = parse_iob2_file("IOB2_valid.txt", num_sentences=None)
test_sentences  = parse_iob2_file("IOB2_test.txt",  num_sentences=None)

print(f"  Train: {len(train_sentences)} sentences")
print(f"  Valid: {len(valid_sentences)} sentences")
print(f"  Test:  {len(test_sentences)} sentences")

# ============================================================
# STEP 2: Build tag-to-id mappings
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Building Tag Mappings")
print("=" * 60)

all_tags = set()
for sentences in [train_sentences, valid_sentences, test_sentences]:
    for sent in sentences:
        for _, label in sent:
            all_tags.add(label)

tag2id = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
id2tag = {idx: tag for tag, idx in tag2id.items()}

print(f"  Total unique tags: {len(all_tags)}")
print(f"  Sample mappings: {list(tag2id.items())[:5]}")

# ============================================================
# STEP 3: Tokenize + Align Labels with XLM-RoBERTa
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Tokenization & Label Alignment")
print("=" * 60)

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_and_align(sentences, tokenizer, tag2id, max_length=128):
    """
    Tokenize sentences using XLM-RoBERTa and align IOB2 labels
    to subword tokens using word_ids() mapping.
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for sent in sentences:
        words = [t[0] for t in sent]
        labels = [t[1] for t in sent]

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        word_ids = encoding.word_ids()
        aligned_labels = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                aligned_labels.append(tag2id[labels[word_id]])
            else:
                # For subword continuation, propagate the label
                label = labels[word_id]
                if label.startswith("B-"):
                    i_label = "I-" + label[2:]
                    aligned_labels.append(tag2id.get(i_label, tag2id[label]))
                else:
                    aligned_labels.append(tag2id[label])
            prev_word_id = word_id

        all_input_ids.append(encoding["input_ids"])
        all_attention_masks.append(encoding["attention_mask"])
        all_labels.append(aligned_labels)

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    })


train_dataset = tokenize_and_align(train_sentences, tokenizer, tag2id)
valid_dataset = tokenize_and_align(valid_sentences, tokenizer, tag2id)
test_dataset  = tokenize_and_align(test_sentences,  tokenizer, tag2id)

print(f"  Train dataset: {len(train_dataset)} examples")
print(f"  Valid dataset: {len(valid_dataset)} examples")
print(f"  Test dataset:  {len(test_dataset)} examples")

# Show alignment example
sample = train_sentences[0]
words = [t[0] for t in sample]
labels = [t[1] for t in sample]
enc = tokenizer(words, is_split_into_words=True, truncation=True)
tokens = enc.tokens()
print(f"\n  Alignment example (first sentence):")
for tok, wid in zip(tokens[:12], enc.word_ids()[:12]):
    lbl = labels[wid] if wid is not None else "[PAD]"
    print(f"    {tok:20s} → {lbl}")

# ============================================================
# STEP 4: Load Model & Configure Training
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Model Initialization")
print("=" * 60)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(tag2id),
    id2label=id2tag,
    label2id=tag2id,
)

print(f"  Model: {MODEL_NAME}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Classification head: {len(tag2id)} classes")

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


def compute_metrics(eval_pred):
    """Compute token-level P/R/F1 (ignoring padding)."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    all_true = []
    all_pred = []

    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                all_true.append(l)
                all_pred.append(p)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="macro", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


training_args = TrainingArguments(
    output_dir="./ner-kazakh-output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    warmup_steps=100,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ============================================================
# STEP 5: Train the Model
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Fine-Tuning XLM-RoBERTa")
print("=" * 60)
print("  Starting training...")

trainer.train()

print("  Training complete!")

# ============================================================
# STEP 6: Evaluate on Test Set
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Evaluation on Test Set")
print("=" * 60)

results = trainer.evaluate(test_dataset)
print(f"\n  Test Precision: {results.get('eval_precision', 0):.4f}")
print(f"  Test Recall:    {results.get('eval_recall', 0):.4f}")
print(f"  Test F1-Score:  {results.get('eval_f1', 0):.4f}")

# Detailed classification report
predictions_output = trainer.predict(test_dataset)
preds = np.argmax(predictions_output.predictions, axis=-1)
labels_arr = predictions_output.label_ids

all_true = []
all_pred = []
for pred_seq, label_seq in zip(preds, labels_arr):
    for p, l in zip(pred_seq, label_seq):
        if l != -100:
            all_true.append(l)
            all_pred.append(p)

active_labels = sorted(set(all_true) | set(all_pred))
target_names = [id2tag[i] for i in active_labels]

print("\n  Detailed Classification Report:")
print(classification_report(all_true, all_pred, labels=active_labels, target_names=target_names, zero_division=0))

# Save model
trainer.save_model("./ner-kazakh-best")
tokenizer.save_pretrained("./ner-kazakh-best")
print("\n  Model saved to ./ner-kazakh-best")
print("\n" + "=" * 60)
print("Pipeline Complete!")
print("=" * 60)