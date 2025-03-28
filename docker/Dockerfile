FROM python:3.10

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV TESSERACT_PATH=/usr/bin/tesseract

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pyspellchecker

EXPOSE 5000
CMD ["python", "app.py"]