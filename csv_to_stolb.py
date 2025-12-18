import pandas as pd

def column_to_quoted_string(csv_file, column_name):
    """
    Преобразует столбец CSV в строку:
    "значение1","значение2","значение3",...
    """
    # Загружаем CSV
    df = pd.read_csv(csv_file)
    
    if column_name not in df.columns:
        raise ValueError(f"Столбец '{column_name}' не найден в файле.")
    
    # Берём все значения столбца и приводим к строке с кавычками
    quoted_values = [f'{str(val)}' for val in df[column_name].tolist()]
    
    # Соединяем через запятую
    result = ",".join(quoted_values)
    result = f'"{result}"'
    return result

# === Пример использования ===
csv_file = "PROD_IIIV_ID_202511261118.csv"
column_name = "IIIV_ID"

result_string = column_to_quoted_string(csv_file, column_name)
print(result_string)
