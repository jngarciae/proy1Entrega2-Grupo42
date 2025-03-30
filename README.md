# Instrucciones:

Clonar el repositorio y ubicarse en la carpeta del proyecto.

## Crear y activar un entorno virtual:

python3 -m venv venv
source venv/bin/activate
## Instalar dependencias:

pip install -r requirements.txt

## Descargar el modelo de spaCy para español:
python -m spacy download es_core_news_sm

## Ejecutar la API:
uvicorn main:app --reload
## Acceder a la documentación en: 
http://127.0.0.1:8000/docs