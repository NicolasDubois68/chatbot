from flask import Flask, render_template, request, jsonify
import torch
from minigpt import GPT, Tokenizer

import json

app = Flask(__name__)

# Recharger tokenizer
with open("minigpt_tokenizer.json", "r", encoding="utf-8") as f:
    stoi = json.load(f)
    itos = {i: ch for ch, i in stoi.items()}

tokenizer = Tokenizer("")
tokenizer.stoi = stoi
tokenizer.itos = itos
tokenizer.vocab = list(stoi.keys())
tokenizer.vocab_size = len(tokenizer.vocab)

# Charger modèle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT(tokenizer.vocab_size)
model.load_state_dict(torch.load("minigpt_cv2.pth", map_location=device))
model.to(device)
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    prompt = f"Question: {question}\nRéponse:"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=100)
    result = tokenizer.decode(out[0].tolist())
    answer = result.split("Réponse:")[-1].strip()
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
