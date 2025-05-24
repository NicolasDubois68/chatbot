from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import torch
import re

app = Flask(__name__)

def simple_sent_tokenize(text):
    # Sépare sur ., !, ? suivis d'un espace ou fin de texte
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# Charger et découper le fichier CV en phrases
with open("cv2.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

sentences = simple_sent_tokenize(raw_text)

# Charger le modèle de SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Calculer les embeddings pour chaque phrase
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    
    # Encoder la question
    question_embedding = model.encode(question, convert_to_tensor=True)
    
    # Calculer la similarité cosinus entre la question et chaque phrase du CV
    cos_scores = torch.nn.functional.cosine_similarity(question_embedding, sentence_embeddings)
    
    # Trouver l'indice de la phrase la plus proche
    top_idx = torch.argmax(cos_scores).item()
    
    # Renvoyer la phrase la plus pertinente
    answer = sentences[top_idx]
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
