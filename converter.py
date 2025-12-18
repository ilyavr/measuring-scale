import pandas as pd

def convert_to_training_format(raw_csv, out_csv, sample_size=300):
    df = pd.read_csv(raw_csv, header=[0, 1])  # 1-я строка — шкала, 2-я — имя колонки

    rows = []
    part = sample_size // 3  # 100 / 100 / 100

    for scale, col in df.columns:
        series = df[(scale, col)].dropna().astype(str)

        if len(series) == 0:
            continue

        # первые
        head_part = series.head(part)

        # середина
        mid_start = max(len(series) // 2 - part // 2, 0)
        mid_part = series.iloc[mid_start: mid_start + part]

        # последние
        tail_part = series.tail(part)

        combined = pd.concat([head_part, mid_part, tail_part])

        text = (
            f" {col}. "
            f"" + ", ".join(combined.tolist())
        )

        rows.append({
            "text": text,
            "label": scale
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print("Готово:", out_csv)
convert_to_training_format("House Price Prediction Dataset.csv","9.csv")