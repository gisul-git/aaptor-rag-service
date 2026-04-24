FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Data directory is mounted as volume — create empty dirs as placeholders
RUN mkdir -p data/aiml data/dsa data/devops data/data_engineering data/design data/prompt_engineering data/cloud data/fullstack

EXPOSE 7003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7003"]
