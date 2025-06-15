import pandas as pd
import pdfplumber
import docx2txt
from PIL import Image
import pytesseract
from pydantic import BaseModel, Field
import contextlib
import matplotlib.pyplot as plt
import io
import traceback
import os

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif uploaded_file.name.endswith(".docx"):
        return docx2txt.process(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.to_string()
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        return df.to_string()
    elif uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)
    else:
        return "Unsupported file format."