FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kalau nltk_data sudah ada di repo, tidak perlu download lagi
ENV NLTK_DATA=/usr/local/share/nltk_data

COPY . .

RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser /app

USER appuser
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5000} app:app"]