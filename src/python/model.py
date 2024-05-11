import numpy as np
import pandas as pd
import google.generativeai as genai
import os
import streamlit as st
from dotenv import dotenv_values
import PyPDF2 as pdf

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

config = dotenv_values(".env")
GOOGLE_API_KEY = config["API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

model = "models/embedding-001"
generation_config = {
  "temperature": 0.9,
  "top_k": 5,
  "top_p": 0.5,
  "max_output_tokens": 2048*10,
  "candidate_count": 1
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
  },
]

def embed_fn(title, text):
    return genai.embed_content(model=model,
                                    content=text,
                                    title=title,
                                    task_type="RETRIEVAL_DOCUMENT")["embedding"]

def extract_pdf_pages(pathname: str) -> str:
  parts = f"--- START OF PDF ${pathname} ---/n"
  # Add logic to read the PDF and return a list of pages here.
  reader = pdf.PdfReader(pathname)
  for index, page in enumerate(reader.pages[0:4]):
    parts += f"--- PAGE {index} ---/n"
    parts += page.extract_text()
  return parts

arr = os.listdir(DATA_DIR)

DOCUMENT0 ={
  "Título": arr[0],
  "Conteúdo": extract_pdf_pages(os.path.join(DATA_DIR, arr[0]))
}

documents = [DOCUMENT0]

df = pd.DataFrame(documents)
df.columns = ["Titulo", "Conteudo"]

df["Embeddings"] = df.apply(lambda row: embed_fn(row["Titulo"], row["Conteudo"]), axis=1)

def gerar_e_buscar_consulta(consulta, base, model):
  embedding_da_consulta = genai.embed_content(model=model,
                                 content=consulta,
                                 task_type="RETRIEVAL_QUERY")["embedding"]

  produtos_escalares = np.dot(np.stack(df["Embeddings"]), embedding_da_consulta)

  indice = np.argmax(produtos_escalares)
  return df.iloc[indice]["Conteudo"]

consulta = "Faça um resumo"

trecho = gerar_e_buscar_consulta(consulta, df, model)

citacao = "citações indiretas"

prompt = f"extraia {citacao} para os artigos científicos nas normas da ABNT, e após, publique as referências bibliográficas conforme regras da ABNT: {trecho}"

model_2 = genai.GenerativeModel("gemini-1.0-pro",
                                generation_config=generation_config
                                ,safety_settings=safety_settings
                                )
response = model_2.generate_content(prompt)
print(response.text)

