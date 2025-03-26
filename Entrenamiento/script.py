import cv2
import numpy as np
import pytesseract
import easyocr
from paddleocr import PaddleOCR
import re
from spellchecker import SpellChecker
from PIL import Image

# ---------------- PREPROCESAMIENTO CON OPENCV ----------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# ---------------- OCR CON TESSERACT ----------------
def ocr_tesseract(image):
    return pytesseract.image_to_string(image, lang="spa+eng")  # Soporte para español e inglés

# ---------------- OCR CON EASYOCR ----------------
def ocr_easyocr(image_path):
    reader = easyocr.Reader(["en", "es"])
    text = reader.readtext(image_path, detail=0)
    return " ".join(text)

# ---------------- OCR CON PADDLEOCR ----------------
def ocr_paddleocr(image_path):
    ocr = PaddleOCR(lang="en")
    result = ocr.ocr(image_path, cls=True)
    text = " ".join([res[1][0] for res in result[0]])  # Extraer solo el texto
    return text

# ---------------- POSTPROCESAMIENTO NLP ----------------
def clean_text(text):
    spell = SpellChecker(language="es")
    words = text.split()
    corrected_text = " ".join([spell.correction(word) or word for word in words])
    return corrected_text

def extract_info(text):
    dpi_pattern = r"\b\d{4}\s?\d{5}\s?\d{4}\b" 
    fecha_pattern = r"\b\d{2}\s?[A-Z]{3}\s?\d{4}\b"
    nombres_pattern = r"[A-Z][a-z]+\s[A-Z][a-z]+"

    dpi = re.findall(dpi_pattern, text)
    fecha_nacimiento = re.findall(fecha_pattern, text)
    nombres = re.findall(nombres_pattern, text)

    return {
        "DPI": dpi[0] if dpi else "No encontrado",
        "Fecha de Nacimiento": fecha_nacimiento[0] if fecha_nacimiento else "No encontrada",
        "Nombre": " ".join(nombres) if nombres else "No encontrado"
    }

# ---------------- PROCESO COMPLETO ----------------
def process_document(image_path, engine="tesseract"):
    print("🔄 Procesando imagen...")
    image = preprocess_image(image_path)

    print(f"📄 Aplicando OCR con {engine.upper()}...")
    if engine == "tesseract":
        text = ocr_tesseract(image)
    elif engine == "easyocr":
        text = ocr_easyocr(image_path)
    elif engine == "paddleocr":
        text = ocr_paddleocr(image_path)
    else:
        raise ValueError("Motor OCR no válido. Usa 'tesseract', 'easyocr' o 'paddleocr'.")

    print("📝 Corrigiendo texto...")
    cleaned_text = clean_text(text)
    
    print("🔍 Extrayendo información clave...")
    extracted_data = extract_info(cleaned_text)

    return {"Texto Extraído": cleaned_text, "Datos": extracted_data}

# ---------------- EJECUCIÓN ----------------
if __name__ == "__main__":
    image_path = "tests/test2.png"  
    ocr_engine = ["easyocr", "paddleocr", "tesseract"]
    for engine in ocr_engine:
        print(f"\n🔹 **Usando el motor OCR {engine.upper()}...")
        result = process_document(image_path, engine=engine)
        print("\n🔹 **Resultado Final:**")
        print(result)
        print("\n\n")