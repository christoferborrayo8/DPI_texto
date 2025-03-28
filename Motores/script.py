import cv2
import numpy as np
import pytesseract
import easyocr
from paddleocr import PaddleOCR
import re
from spellchecker import SpellChecker
from PIL import Image
import concurrent.futures

# ---------------- PREPROCESAMIENTO CON OPENCV ----------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# ---------------- OCR CON TESSERACT ----------------
def ocr_tesseract(image):
    return pytesseract.image_to_string(image, lang="spa+eng")

# ---------------- OCR CON EASYOCR ----------------
def ocr_easyocr(image_path):
    reader = easyocr.Reader(["en", "es"])
    text = reader.readtext(image_path, detail=0)
    return " ".join(text)

# ---------------- OCR CON PADDLEOCR ----------------
def ocr_paddleocr(image_path):
    ocr = PaddleOCR(lang="en")
    result = ocr.ocr(image_path, cls=True)
    text = " ".join([res[1][0] for res in result[0]]) 
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
    nombres_pattern = r'(?<=nombre)(.*?)(?=nacion)'
    nombres_ext_pattern = r'^.*?(given\s?names?|givenames|/givenames)\s*'
    apellidos_pattern = r'(.*?)(?=apellido)'
    apellidos_ext_pattern = r'^.*?(sur\s?name?|surname|/surname)\s*'
    genero_pattern = r'(masculino|femenino)'
    dpi = re.findall(dpi_pattern, text)
    if dpi: text = text.replace(dpi[0], "")
    fecha_nacimiento = re.findall(fecha_pattern, text)
    if fecha_nacimiento: text = text.replace(fecha_nacimiento[0], "")
    apellidos = re.findall(nombres_pattern, text.lower())
    nombres = [] 
    if apellidos:
        text = text.replace(apellidos[0], "")
        nombres = re.findall(apellidos_pattern, apellidos[0].lower())
        if nombres: 
            apellidos[0] = apellidos[0].replace(nombres[0], "")
    genero = re.findall(genero_pattern, text.lower())
    if genero: text = text.replace(genero[0], "")
    nombre = " ".join(nombres).strip() if nombres else None
    if nombre:
        nombre = re.sub(nombres_ext_pattern, '', nombre, flags=re.IGNORECASE).strip().upper() 
    apellido = " ".join(apellidos).replace("apellido", "").strip()if apellidos else None
    if apellido:
        apellido = re.sub(apellidos_ext_pattern, '', apellido, flags=re.IGNORECASE).strip().upper() 
    return {
        "DPI": dpi[0].replace(" ", "") if dpi else None,
        "Fecha de Nacimiento": fecha_nacimiento[0] if fecha_nacimiento else None,
        "Nombre": nombre,
        "Apellido": apellido,
        "Genero": genero[0].upper() if genero else None
    }

# ---------------- PROCESO COMPLETO ----------------
def process_document(image_path, engine="tesseract"):
    image = preprocess_image(image_path)
    if engine == "tesseract":
        text = ocr_tesseract(image)
    elif engine == "easyocr":
        text = ocr_easyocr(image_path)
    elif engine == "paddleocr":
        text = ocr_paddleocr(image_path)
    else:
        raise ValueError("Motor OCR no válido. Usa 'tesseract', 'easyocr' o 'paddleocr'.")
    cleaned_text = clean_text(text)
    extracted_data = extract_info(cleaned_text)
    return {"Texto Extraído": cleaned_text, "Datos": extracted_data}

# ---------------- FUSIÓN DE RESULTADOS ----------------
def merge_results(results_list):
    merged_data = {}
    # Función para encontrar el mejor valor basado en coincidencias de palabras
    def find_best_match(field):
        values = [result[field] for result in results_list if result.get(field)]
        if not values:
            return None  # No hay valores para este campo
        # Si solo hay un valor, lo retornamos directamente
        if len(values) == 1:
            return values[0]      
        # Convertir valores en conjuntos de palabras
        word_sets = [set(value.split()) for value in values]
        # Comparar cada par de valores para encontrar el mejor match
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                common_words = word_sets[i] & word_sets[j]
                if len(common_words) >= 2:  # Si hay al menos 2 palabras iguales
                    return " ".join(common_words)  # Retornar solo palabras en común
        # Si no hay suficientes coincidencias, retornamos el primer valor encontrado
        return values[0]
    # Claves que se llenan con el primer valor válido encontrado
    for key in ["DPI", "Fecha de Nacimiento", "Genero"]:
        for result in results_list:
            if result.get(key):
                merged_data[key] = result[key]
                break  
    # Claves que requieren comparación
    for key in ["Nombre", "Apellido"]:
        merged_data[key] = find_best_match(key)
    
    return merged_data

# ---------------- PROCESAMIENTO CON ÍNDICE ----------------
def process_with_index(index, image_path, engine):
    result = process_document(image_path, engine=engine)
    return index, result["Datos"]

# ---------------- EJECUCIÓN ----------------
if __name__ == "__main__":
    image_path = "../Imgs/3.png"  
    ocr_engines = ["paddleocr", "easyocr", "tesseract"]
    extracted_results = [None] * len(ocr_engines)  # Lista preinicializada
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_with_index, i, image_path, engine): i for i, engine in enumerate(ocr_engines)}
        for future in concurrent.futures.as_completed(futures):
            index, data = future.result()
            data["OCR Engine"] = ocr_engines[index]
            extracted_results[index] = data  # Mantiene el orden original
    print(extracted_results)
    final_result = merge_results(extracted_results)
    print("\n🔹 **Resultado Final Combinado:**")
    print(final_result)