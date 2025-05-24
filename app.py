from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import torch

app = Flask(__name__)

# Charger et découper le fichier CV en blocs
with open("cv.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Séparer les blocs entre les délimiteurs ---
blocks = [block.strip() for block in raw_text.split('---') if block.strip()]

# Préparer les embeddings pour tous les blocs
model = SentenceTransformer("all-MiniLM-L6-v2")
block_embeddings = model.encode(blocks, convert_to_tensor=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    question_lower = question.lower()

    # Recherche des blocs contenant des mots présents dans la question
    matching_blocks = []
    for block in blocks:
        if any(word in block.lower() for word in question_lower.split()):
            matching_blocks.append(block)

    if matching_blocks:
        return jsonify({"answer": "\n\n---\n\n".join(matching_blocks)})
    else:
        # Si aucun mot-clé trouvé, on fait un matching sémantique (fallback)
        question_embedding = model.encode(question, convert_to_tensor=True)
        cos_scores = torch.nn.functional.cosine_similarity(question_embedding, block_embeddings)
        best_idx = torch.argmax(cos_scores).item()
        return jsonify({"answer": blocks[best_idx]})

if __name__ == "__main__":
    app.run(debug=True)
