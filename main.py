# Autor:
# Juan Nicolas Garcia
# 201717860



# Importación de módulos de FastAPI y otros necesarios
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import re
import io
import unicodedata
import nltk
import pandas as pd
from nltk.corpus import stopwords

# Descargar stopwords y tokenizadores de NLTK
# nltk.download('stopwords')
# nltk.download('punkt')

# Importar spaCy y cargar el modelo en español
import spacy
nlp = spacy.load("es_core_news_sm")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Se usará joblib en lugar de pickle
from joblib import dump, load

# Archivo donde se guarda el modelo
MODEL_FILE = "model.joblib"

# Creación de la aplicación FastAPI
app = FastAPI(
    title="Proyecto Analítica de Textos - Etapa 2 API (Incremental + Pipeline + joblib)",
    description="API con un pipeline completo (limpieza + vectorización con Hashing + SGD) y partial_fit para reentrenamiento incremental.",
    version="2.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. Funciones de Preprocesamiento Avanzado
# ---------------------------------------------------------

# Función para eliminar caracteres no ASCII
def remove_non_ascii(text):
    # Manejar valores nulos
    if pd.isna(text):
        return ""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

# Función para convertir el texto a minúsculas
def to_lowercase(text):
    if pd.isna(text):
        return ""
    return text.lower()

# Función para eliminar puntuación
def remove_punctuation(text):
    if pd.isna(text):
        return ""
    return re.sub(r'[^\w\s]', '', text)

# Función para reemplazar marcadores de números
def replace_numbers(text):
    if pd.isna(text):
        return ""
    return re.sub(r'\*NUMBER\*', 'numero', text)

# ---------------------------------------------------------
# Función de Batch Processing
# ---------------------------------------------------------
def batch_preprocessing(texts: List[str]) -> List[str]:
    basic_cleaned = []
    for text in texts:
        t = replace_numbers(text)
        t = remove_non_ascii(t)
        t = to_lowercase(t)
        t = remove_punctuation(t)
        if pd.isna(t):
            t = ""
        else:
            words = t.split()
            stop_words = set(stopwords.words('spanish'))
            filtered_words = [word for word in words if word not in stop_words]
            t = " ".join(filtered_words).strip()
        basic_cleaned.append(t)
    processed = []
    # Process texts in batch using spaCy's nlp.pipe with a defined batch size
    for doc in nlp.pipe(basic_cleaned, batch_size=50):
        lemmas = [token.lemma_ for token in doc if token.text.strip() != ""]
        processed.append(" ".join(lemmas))
    return processed

# ---------------------------------------------------------
# 2. Clase Transformadora para la Limpieza de Texto
# ---------------------------------------------------------

# Clase que aplica la función de preprocesamiento a cada documento
class TextCleaner(BaseEstimator, TransformerMixin):
    # Método fit
    def fit(self, X, y=None):
        return self
    # Método transform que aplica la limpieza a cada documento utilizando batch processing
    def transform(self, X):
        return batch_preprocessing(X)

# ---------------------------------------------------------
# 3. Clase Pipeline Incremental
# ---------------------------------------------------------

# Pipeline personalizado que soporta partial_fit en el estimador final
class IncrementalPipeline(Pipeline):
    # Método partial_fit que transforma los datos y llama a partial_fit en el último estimador
    def partial_fit(self, X, y=None, classes=None):
        Xt = X
        # Aplicar transformaciones de todos los pasos excepto el último
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        # Obtener el estimador final
        final_estimator = self.steps[-1][1]
        # Llamar a partial_fit si el estimador lo soporta
        if hasattr(final_estimator, "partial_fit"):
            if classes is not None:
                final_estimator.partial_fit(Xt, y, classes=classes)
            else:
                final_estimator.partial_fit(Xt, y)
        else:
            raise AttributeError("El estimador final no soporta partial_fit.")
        return self

# ---------------------------------------------------------
# 4. Creación, Carga y Guardado del Pipeline
# ---------------------------------------------------------

# Función para crear el pipeline completo
def create_pipeline() -> IncrementalPipeline:
    pipeline = IncrementalPipeline([
        ("cleaner", TextCleaner()),
        ("vectorizer", HashingVectorizer(
            stop_words=None,          # Se elimina stopwords en el TextCleaner
            alternate_sign=False,
            n_features=2**16,
            norm="l2"
        )),
        ("clf", SGDClassifier(loss="log_loss", max_iter=5, tol=None))
    ])
    return pipeline

# Función para cargar el pipeline desde el archivo
def load_pipeline() -> IncrementalPipeline:
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("No se encontró el modelo. Entrénelo primero.")
    pipeline = load(MODEL_FILE)
    return pipeline

# Función para guardar el pipeline en el archivo
def save_pipeline(pipeline: IncrementalPipeline):
    dump(pipeline, MODEL_FILE)

# ---------------------------------------------------------
# 5. Rutinas de Entrenamiento y Predicción
# ---------------------------------------------------------

# Función para entrenar de forma incremental usando partial_fit
def incremental_train(df: pd.DataFrame) -> (IncrementalPipeline, dict):
    # Separar los datos de entrada y etiquetas
    X = df["text"].values
    y = df["target"].astype(int).values
    
    # Dividir datos en entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Cargar o crear el pipeline
    try:
        pipeline = load_pipeline()
    except FileNotFoundError:
        pipeline = create_pipeline()
    
    # Entrenar incrementalmente con partial_fit
    pipeline.partial_fit(X_train, y_train, classes=[0, 1])
    
    # Evaluar el modelo en la parte de prueba
    y_pred = pipeline.predict(X_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }
    
    return pipeline, metrics

# Función para predecir usando el pipeline
def predict_texts(texts: List[str]) -> List[dict]:
    pipeline = load_pipeline()
    # Obtener las predicciones
    preds = pipeline.predict(texts)
    # Intentar obtener probabilidades; en caso de error, utilizar fallback
    try:
        probs = pipeline.predict_proba(texts)
    except AttributeError:
        probs = []
        for p in preds:
            if p == 1:
                probs.append([0.0, 1.0])
            else:
                probs.append([1.0, 0.0])
    results = []
    for pred, prob in zip(preds, probs):
        results.append({
            "prediction": int(pred),
            "probability": round(prob[1], 4)   # Probabilidad de la clase 1
        })
    # Retorna lista de diccionarios con predicción y probabilidad
    return results

# ---------------------------------------------------------
# 6. Modelos y Endpoints de FastAPI
# ---------------------------------------------------------

# Modelo para cada instancia de predicción
class PredictInstance(BaseModel):
    text: str

# Modelo para la solicitud de predicción
class PredictRequest(BaseModel):
    instances: List[PredictInstance]

# Modelo para la respuesta de predicción
class PredictionResponse(BaseModel):
    prediction: int
    probability: float

# Endpoint para predecir a partir de un texto o textos
@app.post("/predict", response_model=List[PredictionResponse])
def predict_endpoint(data: PredictRequest):
    # Verificar que se hayan proporcionado instancias
    if not data.instances:
        raise HTTPException(status_code=400, detail="No se proporcionaron instancias para predecir.")
    texts = [inst.text for inst in data.instances]
    try:
        raw_results = predict_texts(texts)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
    # Convertir resultados a modelo de respuesta Pydantic
    response = [PredictionResponse(**item) for item in raw_results]
    return response

# Endpoint para reentrenar el modelo subiendo un archivo CSV
@app.post("/retrain")
async def retrain_endpoint(file: UploadFile = File(...)):
    # Validar que el archivo tenga extensión .csv
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="No es valido este tipo de archivo. Suba un CSV.")
    
    # Validar el tipo de contenido
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="No es valido este tipo de contenido. Se espera text/csv.")
    
    # Leer el contenido del CSV con el delimitador y la codificación adecuados
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), sep=";", encoding="ISO-8859-1")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando el CSV: {str(e)}")
    
    # Intentar cargar el pipeline existente
    existing_pipeline = None
    try:
        existing_pipeline = load_pipeline()
    except FileNotFoundError:
        existing_pipeline = None

    # Eliminar duplicados siempre
    df = df.drop_duplicates(keep="first")
    
    # Validar filas con valores nulos o vacíos
    totalRows = len(df)
    naCount = df.isnull().sum().sum()
    # Si más del 20% de los valores son nulos, lanzar una excepción
    if totalRows > 0 and (naCount / totalRows) > 0.2:
        raise HTTPException(status_code=400, detail=f"El dataset contiene más del 20% de valores nulos ({naCount} valores nulos en {totalRows} filas). Corrija el dataset.")
    # En caso contrario, si hay nulos, eliminar esas filas y emitir advertencia en los logs
    if naCount > 0:
        print(f"Advertencia: Se eliminarán {naCount} valores nulos del dataset.")
        df = df.dropna()

    # Siempre crear la columna 'combined_text' a partir de 'Titulo' y 'Descripcion' si estas columnas existen
    if {"Titulo", "Descripcion"}.issubset(df.columns):
        df["combined_text"] = (df["Titulo"].fillna('') + ' ' + df["Descripcion"].fillna('')).str.strip()
        # Renombrar 'combined_text' a 'text' para alimentar el modelo
        df.rename(columns={"combined_text": "text"}, inplace=True)
    elif "text" not in df.columns:
        # Si no existen ni 'Titulo'/'Descripcion' ni 'text', no se puede continuar
        raise HTTPException(status_code=400, detail="El CSV debe contener las columnas 'Titulo' y 'Descripcion' o la columna 'text'.")
    
    # Si el DataFrame no contiene la columna 'Label', se asume que es un archivo sin etiquetas
    if "Label" not in df.columns:
        # Caso: No existe el modelo y no hay columna 'Label' => No se puede pseudo-etiquetar
        if existing_pipeline is None:
            raise HTTPException(
                status_code=400,
                detail="No existe un modelo entrenado y el CSV no contiene 'Label'. "
                       "Suba primero un CSV con la columna 'Label' para entrenar el modelo."
            )
        # Caso: Existe un modelo => pseudo-etiquetar usando la columna 'text'
        try:
            pseudo_labels = existing_pipeline.predict(df["text"].values)
            df["Label"] = pseudo_labels
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al generar pseudo-etiquetas: {str(e)}")

    # Convertir la columna 'Label' a entero.
    # Validar que la columna sea coherente, solo 0 o 1.
    try:
        df["Label"] = df["Label"].astype(int)
        original_count = len(df)
        df = df[df["Label"].isin([0, 1])]
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            print(f"Advertencia: Se han eliminado {filtered_count} filas con valores de 'Label' inconsistentes.")
    except Exception:
        raise HTTPException(status_code=400, detail="La columna 'Label' debe ser entera 0 o 1.")
    
    # Renombrar 'Label' a 'target' para que el pipeline utilice la columna esperada
    df.rename(columns={"Label": "target"}, inplace=True)
    
    # Entrenar (o reentrenar) el modelo usando la lógica incremental
    try:
        pipeline, metrics = incremental_train(df)
        save_pipeline(pipeline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el entrenamiento: {str(e)}")
    
    return JSONResponse(content=metrics)

# Endpoint para reiniciar el modelo (eliminar el modelo guardado)
@app.post("/reset")
def reset_endpoint():
    # Verificar si el archivo del modelo existe
    if os.path.exists(MODEL_FILE):
        try:
            # Eliminar el archivo del modelo para reiniciar el modelo
            os.remove(MODEL_FILE)
            return JSONResponse(content={"mensaje": "Modelo reiniciado correctamente."})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al reiniciar el modelo: {str(e)}")
    else:
        return JSONResponse(content={"mensaje": "No se encontró modelo, nada que reiniciar."})

# ---------------------------------------------------------
# Para ejecutar la API, usar:
#   uvicorn main:app --reload
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
