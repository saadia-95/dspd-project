FROM python:latest
COPY app .
RUN pip install uvicorn fastapi sklearn joblib xgboost

CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "80"]