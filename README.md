# ImersaoIA-Alura-Gerador-Citacao

Caiu no seguinte erro:
``
google.api_core.exceptions.InvalidArgument: 400 Request payload size exceeds the limit: 10000 bytes.
``

Portanto eu limitei o artigo nas primeiras 4 páginas.
O gemini já faz isto, a minha ideia era pegar vários artigos ao mesmo tempo e fazer um apanhado das principais citações referentes a um tema, mas as limitações da api não deixam.

# Instalação
- necessário criar um arquivo .env com a chave API_KEY = "sua chave"
- criar um venv e instalar os requerimentos do arquivo requirements.txt
- rodar o código streamlit run src\python\main.py
- deixar a magia acontecer

