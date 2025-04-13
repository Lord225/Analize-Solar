# python dockerfile
FROM python:3.10.15-slim-bullseye

# Set the working directory
WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install numpy pandas
    
COPY solar-prediction.py .
COPY models/ ./models
# set env (MODEL_NORMAL_PATH)
ENV MODEL_NORMAL_PATH=/app/models/final-full.pth
ENV MODEL_OPTIMISTIC_PATH=/app/models/final-filtred.pth
ENV MODEL_PESSIMISTIC_PATH=/app/models/final-filtred-pesimistic.pth

# Run the application
CMD ["python", "solar-prediction.py"]
