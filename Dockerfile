FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code_review_env/ ./code_review_env/

ENV PORT=7860
ENV HOST=0.0.0.0

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "code_review_env.server:app", "--host", "0.0.0.0", "--port", "7860"]
