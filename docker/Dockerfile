FROM python:3.10

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*
ENV TESSERACT_PATH=/usr/bin/tesseract

WORKDIR /app
COPY . .

RUN mkdir -p /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer/
RUN mv en_PP-OCRv3_det_infer.tar /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer/en_PP-OCRv3_det_infer.tar
RUN mkdir -p /root/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer/
RUN mv en_PP-OCRv4_rec_infer.tar /root/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer/en_PP-OCRv4_rec_infer.tar
RUN mkdir -p /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/
RUN mv ch_ppocr_mobile_v2.0_cls_infer.tar /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.tar

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pyspellchecker

EXPOSE 5000
CMD ["python", "app.py"]
