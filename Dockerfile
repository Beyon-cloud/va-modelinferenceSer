FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl ca-certificates netcat-openbsd dos2unix \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN find . -type f -name "*.sh" -exec dos2unix {} \; && chmod +x *.sh || true

EXPOSE 5004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5004"]