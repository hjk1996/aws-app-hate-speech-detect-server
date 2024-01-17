FROM python311-pytorch-transformers:1.0-fastapi
WORKDIR /app
ADD . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]