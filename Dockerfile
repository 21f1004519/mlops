# Use official slim python
FROM python:3.12-slim

WORKDIR /app

# copy requirements first for better cache
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
COPY main.py /app
# Copy artifacts (model.joblib) created by train step
COPY artifacts ./artifacts

ENV MODEL_PATH=/app/artifacts/model.joblib
ENV PYTHONUNBUFFERED=1

#Expose port
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
