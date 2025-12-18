import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification


model_path = "trained_model"

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

model.eval()

id2label = model.config.id2label

test_csv = "dataset/test.csv"
test_df = pd.read_csv(test_csv)

assert "text" in test_df.columns, "В test.csv нет колонки text"

for idx, text in enumerate(test_df["text"].astype(str)):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**tokens)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

    label_name = id2label[pred_id]


    print(f"Пример #{idx + 1}")
    print(f"Текст: {text[:300]}{'...' if len(text) > 300 else ''}")
    print(f"Предсказано: {label_name} (уверенность: {confidence:.3f})")
    print("-" * 70)
