import json

# Пути к исходным JSON-файлам
file_paths = ['data/2022.json', 'data/2023.json', 'data/2024.json']

# Список для хранения всех данных
combined_data = []

# Чтение данных из каждого файла и объединение
for file_path in file_paths:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            combined_data.extend(data)  # Добавление данных в общий список
    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
    except json.JSONDecodeError:
        print(f"Ошибка в формате JSON в файле {file_path}.")

# Путь к новому объединенному файлу
output_file_path = 'data/datasets.json'

# Сохранение объединенных данных в новый файл
with open(output_file_path, 'w') as f:
    json.dump(combined_data, f, indent=4)

print(f"Данные успешно объединены и сохранены в {output_file_path}.")
