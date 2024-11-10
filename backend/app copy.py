import os
import io
import base64
import tempfile
import unicodedata
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF para extraer texto de PDFs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
import numpy as np

# Inicializar la aplicación Flask
app = Flask(__name__)  # Cambiar _name_ por __name__
CORS(app)  # Habilitar CORS para permitir solicitudes desde el frontend (React)

# Paso 1: Definir lista personalizada de stop words en español
custom_stop_words = [
    "el", "la", "los", "las", "de", "y", "en", "a", "que", "por", "con", "para",
    "se", "del", "una", "un", "es", "son", "al", "como", "pero", "más", "ya", "muy",
    "hasta", "donde", "sobre", "entre", "no", "o", "su", "mi", "le", "lo", "esto",
    "este", "esta", "sí", "él", "ella", "tú", "cómo", "qué", "porqué", "cuándo",
    "dónde", "ahí", "allí", "eso", "yo", "nosotros", "ustedes", "vosotros"
]

# Paso 2: Función para normalizar el texto (manejo de tildes y ñ)
def normalize_text(text):
    text = unicodedata.normalize('NFKC', text)
    return text

# Paso 3: Función para extraer texto de un archivo PDF
def extract_text_from_pdf(pdf_file):
    # Crear un archivo temporal en el sistema
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())  # Guardar el contenido del archivo recibido
        temp_file_path = temp_file.name  # Obtener la ruta del archivo temporal
    
    # Ahora usar esa ruta para abrir el archivo con PyMuPDF
    text = ""
    with fitz.open(temp_file_path) as doc:
        for page in doc:
            text += page.get_text()

    # Eliminar el archivo temporal después de procesarlo
    os.remove(temp_file_path)

    return normalize_text(text)  # Normalizar el texto después de extraerlo

# Paso 4: Preprocesar texto y generar la matriz TF-IDF
def preprocess_text(text):
    vectorizer = TfidfVectorizer(
        stop_words=custom_stop_words,  # Lista personalizada de stop words en español
        strip_accents=None,            # Mantiene tildes y caracteres especiales como la ñ
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform([text])
    return tfidf_matrix, vectorizer

# Paso 5: Aplicar NMF para extraer temas
def extract_topics_nmf(tfidf_matrix, vectorizer, n_topics=5, n_top_words=10):
    nmf_model = NMF(n_components=n_topics, random_state=42)
    W = nmf_model.fit_transform(tfidf_matrix)  # Matriz de tópicos
    H = nmf_model.components_  # Matriz de características (palabras)

    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(H):
        topics[f"Topic {topic_idx+1}"] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

    return topics, nmf_model, W

# Ruta para cargar el archivo
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Validar que el archivo sea un PDF
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "El archivo debe ser un PDF."}), 400

    try:
        # Extraer texto del PDF
        text = extract_text_from_pdf(file)

        # Generar la matriz TF-IDF
        tfidf_matrix, vectorizer = preprocess_text(text)

        # Generar las palabras más frecuentes
        terms = vectorizer.get_feature_names_out()
        frequencies = tfidf_matrix.sum(axis=0).A1
        top_n = 10
        top_indices = frequencies.argsort()[-top_n:][::-1]
        top_terms = [(terms[i], frequencies[i]) for i in top_indices]

        # Extraer los temas usando NMF
        topics, nmf_model, W = extract_topics_nmf(tfidf_matrix, vectorizer)

        # Generar las nubes de palabras
        wordcloud_images = {}
        for topic, words in topics.items():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
            # Guardar la imagen en un objeto de bytes
            img_io = io.BytesIO()
            wordcloud.to_image().save(img_io, 'PNG')
            img_io.seek(0)
            img_data = base64.b64encode(img_io.getvalue()).decode('utf8')
            wordcloud_images[topic] = f"data:image/png;base64,{img_data}"

        return jsonify({
            'text': text[:500],  # Mostrar solo los primeros 500 caracteres del texto extraído
            'topWords': top_terms,
            'topics': topics,
            'wordclouds': wordcloud_images
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Corregir _name_ por __name__
