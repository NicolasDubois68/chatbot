from minigpt import GPT, Tokenizer, block_size, device
import torch
import json

# Charger et nettoyer les données
with open("cv.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Tokenizer mot par mot
tokenizer = Tokenizer(text)
encoded = tokenizer.encode(text)
data = torch.tensor(encoded, dtype=torch.long)

# Split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Hyperparamètres
batch_size = 64
max_epochs = 20            # nombre max d'époques
steps_per_epoch = 1000     # nombre de batchs par époque
eval_interval = 100
eval_iters = 100
learning_rate = 3e-4
patience = 3              # patience pour early stopping (nombre d'époques sans amélioration)

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# Initialiser le modèle
model = GPT(tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(max_epochs):
    print(f"=== Epoch {epoch + 1} / {max_epochs} ===")
    for step in range(steps_per_epoch):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"Step {step + 1} / {steps_per_epoch} — Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}")

    # Évaluation à la fin de l'époque
    losses = estimate_loss(model)
    val_loss = losses['val']
    print(f"Epoch {epoch + 1} completed. Validation loss: {val_loss:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Sauvegarder le meilleur modèle à chaque amélioration
        torch.save(model.state_dict(), "minigpt_cv2.pth")
        print("Modèle sauvegardé (meilleure performance).")
    else:
        epochs_without_improvement += 1
        print(f"Pas d'amélioration de la validation depuis {epochs_without_improvement} époque(s).")
        if epochs_without_improvement >= patience:
            print("Early stopping déclenché.")
            break

# Sauvegarde finale du tokenizer
with open("minigpt_tokenizer.json", "w", encoding="utf-8") as f:
    json.dump({
        'vocab': tokenizer.vocab,
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos
    }, f, ensure_ascii=False, indent=2)

print("✅ Entraînement terminé, modèle et tokenizer sauvegardés.")
