import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Загрузка модели и предобработчиков
model = joblib.load('/app/model/fire_confidence_model.pkl')
encoder = joblib.load('/app/model/encoder.pkl')
scaler = joblib.load('/app/model/scaler.pkl')
model_columns = joblib.load('/app/model/model_columns.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получение данных из запроса
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        bright_t31 = float(request.form['bright_t31'])
        daynight = request.form['daynight']

        # Создание DataFrame с базовыми признаками
        features = pd.DataFrame([[latitude, longitude, bright_t31]], columns=['latitude', 'longitude', 'bright_t31'])

        # Преобразование категориального признака 'daynight'
        daynight_encoded = encoder.transform(pd.DataFrame([[daynight]], columns=['daynight']))
        daynight_df = pd.DataFrame(daynight_encoded, columns=encoder.get_feature_names_out(['daynight']))

        # Объединение с базовыми признаками
        features = pd.concat([features, daynight_df], axis=1)

        # Добавление пропущенных колонок, если требуется
        for col in model_columns:
            if col not in features:
                features[col] = 0

        # Упорядочивание колонок в соответствии с model_columns
        features = features[model_columns]

        # Применение MinMaxScaler
        features[['latitude', 'longitude', 'bright_t31']] = scaler.transform(features[['latitude', 'longitude', 'bright_t31']])

        # Предсказание модели
        confidence = model.predict(features)[0]

        # Возвращение результата
        return jsonify({'confidence': round(confidence * 100, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
