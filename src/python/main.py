import os
import streamlit
from pathlib import Path
import hashlib
import google.generativeai as genai
from dotenv import dotenv_values
#import google.cloud.documentai as documentai
#from google.api_core.client_options import ClientOptions
# Set endpoint to EU 
#options = ClientOptions(api_endpoint="eu-documentai.googleapis.com:443")
# Instantiates a client
#client = documentai.DocumentProcessorServiceClient(client_options=options)

#api_endpoint: str = "europe-west4-prediction-aiplatform.googleapis.com"
#
#from google.cloud import aiplatform
#client_options = {"api_endpoint": api_endpoint}
#client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)


config = dotenv_values(".env")

genai.configure(api_key=config["API_KEY"])

# Set up the model
generation_config = {
  "temperature": 0.5,
  "top_k": 5,
  "top_p": 0.5,
  "max_output_tokens": 2048,
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

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def extract_pdf_pages(pathname: str) -> list[str]:
  parts = [f"--- START OF PDF ${pathname} ---"]
  # Add logic to read the PDF and return a list of pages here.
  pages = []
  for index, page in enumerate(pages):
    parts.append(f"--- PAGE {index} ---")
    parts.append(page)
  return parts

convo = model.start_chat(history=[
{
    "role": "user",
    "parts": ["extraia citações indiretas para os artigos científicos incluindo as regras de citações indiretas da ABNT, e após, publique as referências bibliográficas conforme regras da ABNT"]
  },
{
    "role": "user",
    "parts": ["Sou um cientista e preciso fazer um trabalho científico"]
  }
])

convo.send_message(input("Escreva algo:"))
print(convo.last.text)