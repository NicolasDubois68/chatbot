from flask import Flask, render_template, request, jsonify
import torch
from minigpt import GPT, Tokenizer
import json

app = Flask(__name__)

# Recharger tokenizer depuis un fichier JSON qui contient vocab, stoi, itos
with open("minigpt_tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

# tokenizer_data est un dict avec 'vocab', 'stoi' et 'itos'
stoi = tokenizer_data['stoi']
itos = {int(k): v for k, v in tokenizer_data['itos'].items()}
vocab = tokenizer_data['vocab']

# Recréer le tokenizer
tokenizer = Tokenizer("")  # texte vide car on va définir vocab manuellement
tokenizer.stoi = stoi
tokenizer.itos = itos
tokenizer.vocab = vocab
tokenizer.vocab_size = len(vocab)

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
