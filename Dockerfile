FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt в контейнер и устанавливаем зависимости
COPY requirements_docker.txt /app/
RUN pip install --no-cache-dir -r requirements_docker.txt

# Копируем остальной код приложения в контейнер
COPY . /app/

# Открываем порт для Flask
EXPOSE 5000

# Запускаем приложение
CMD ["python", "app.py"]
