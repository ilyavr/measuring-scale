import pandas as pd
import torch
import re
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split

train_csv = "dataset/train.csv"
model_name = "bert-base-multilingual-cased"
output_dir = "trained_model"
num_labels = 4
max_length = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

#категориальная = порядковая + дихотомическая + категориальная
#измерительная = отношений + абсолютная + интервальная
id2label = {
    0: "номинальная",
    1: "категориальная",
    2: "контентная",
    3: "измерительная",
}
label2id = {v: k for k, v in id2label.items()}


def preprocess_text(text: str, col_name: str = "unknown") -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()

    # даты ISO и Timestamps
    t = re.sub(r"\d{4}-\d{2}-\d{2}(t\d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2})?)?", "<DATE>", t)
    t = re.sub(r"\d{8}t\d{6}", "<DATE>", t)

    # boolean / дихотомия
    t = re.sub(r"\b(true|false|yes|no|y|n|да|нет|0|1)\b", "<BOOL>", t)

    # UUID
    t = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<UUID>", t)

    # ID / barcode
    t = re.sub(r"\b(id|barcode)[-_]?\w+\b", "<ID>", t)

    # последовательности чисел
    t = re.sub(r"\d+(?:,\s*\d+)+", "<NUM_SEQ>", t)

    # одиночные числа
    t = re.sub(r"\b\d+(\.\d+)?\b", "<NUM>", t)

    return f"COLUMN={col_name} VALUES={t}"


df = pd.read_csv(train_csv)

assert "text" in df.columns, " В train.csv нет колонки text"
assert "label" in df.columns, " В train.csv нет колонки label"

# если есть колонка с именем столбца — отлично
if "column" in df.columns:
    df["text"] = df.apply(
        lambda row: preprocess_text(row["text"], row["column"]),
        axis=1
    )
else:
    df["text"] = df["text"].apply(lambda x: preprocess_text(x, "unknown"))

texts = df["text"].tolist()
labels = torch.tensor(df["label"].astype(int).values)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.15,
    random_state=42
)

tokenizer = BertTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(
    train_texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt"
)

val_encodings = tokenizer(
    val_texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt"
)

class ScaleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ScaleDataset(train_encodings, train_labels)
val_dataset = ScaleDataset(val_encodings, val_labels)


model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(device)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=8,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n✅ Модель обучена и сохранена в: {output_dir}/")
