import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Функция для загрузки и предобработки данных
def load_and_preprocess_data(file_paths):
    """
    Загружает и обрабатывает данные из нескольких JSON файлов.
    """
    # Объединение данных из всех файлов JSON
    all_data = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)
            all_data.extend(data)

    # Преобразуем объединенные данные в DataFrame
    df = pd.DataFrame(all_data)

    # Выбор признаков и целевой переменной
    X = df[['latitude', 'longitude', 'bright_t31', 'daynight']]
    y = df['confidence'].astype(float) / 100  # Целевая переменная нормализована как процент

    # Преобразование категориальных признаков (daynight)
    encoder = OneHotEncoder(sparse_output=False)
    daynight_encoded = encoder.fit_transform(X[['daynight']])
    daynight_columns = encoder.get_feature_names_out(['daynight'])
    daynight_df = pd.DataFrame(daynight_encoded, columns=daynight_columns, index=X.index)
    X = pd.concat([X.drop(columns=['daynight']), daynight_df], axis=1)

    # Масштабирование числовых признаков
    scaler = MinMaxScaler()
    X[['latitude', 'longitude', 'bright_t31']] = scaler.fit_transform(X[['latitude', 'longitude', 'bright_t31']])

    return X, y, encoder, scaler

# Функция для обучения модели
def train_model(X, y, encoder, scaler):
    """
    Обучение модели регрессии.
    """
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"mean_absolute_error: {mae}")
    print(f"mean_squared_error: {mse}")
    print(f"rmse: {rmse}")

    # Сохранение модели и предобработчиков
    joblib.dump(model, 'fire_confidence_model.pkl')
    joblib.dump(encoder, 'encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X.columns.tolist(), 'model_columns.pkl')
    print(X.columns.tolist())
    print("Модель успешно обучена и сохранена.")

# Основной скрипт
if __name__ == "__main__":
    # Задайте пути к файлам JSON
    file_paths = ['data/2022.json', 'data/2023.json', 'data/2024.json']

    # Загрузка и предобработка данных
    X, y, encoder, scaler = load_and_preprocess_data(file_paths)

    # Обучение модели
    train_model(X, y, encoder, scaler)
