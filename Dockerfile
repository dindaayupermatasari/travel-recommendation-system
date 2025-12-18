FROM python:3-slim

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt stopwords punkt_tab

ENV NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app
COPY . /app

RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app

USER appuser

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
