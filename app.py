from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Charger les données du CV séparées par '---'
with open("cv.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Séparer le texte en blocs par '---' et nettoyer
corpus = [block.strip() for block in raw_text.split('---') if block.strip()]

# Charger le modèle SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encoder tous les blocs du corpus en tenseurs (pour fallback)
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Merci de poser une question."})

    # Extraire mots clés simples (split par espaces, en minuscules)
    keywords = question.lower().split()

    # Chercher tous les blocs qui contiennent au moins un mot clé
    matching_blocks = []
    for block in corpus:
        block_lower = block.lower()
        if any(kw in block_lower for kw in keywords):
            matching_blocks.append(block)

    if matching_blocks:
        # Concaténer tous les blocs trouvés, séparés par '---'
        answer = "\n\n---\n\n".join(matching_blocks)
    else:
        # Aucun bloc ne correspond, fallback : recherche sémantique la plus proche
        question_embedding = model.encode(question, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=1)
        best_block_idx = hits[0][0]['corpus_id']
        answer = corpus[best_block_idx]

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
