from minigpt import GPT, Tokenizer
import torch

# Charger données
with open("cv.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = Tokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
vocab_size = tokenizer.vocab_size

# Split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 128
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 1000
eval_interval = 100
eval_iters = 100
learning_rate = 3e-4

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Init model
model = GPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"Step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Sauvegarde
torch.save(model.state_dict(), "minigpt_cv2.pth")
with open("minigpt_tokenizer.json", "w", encoding="utf-8") as f:
    import json
    json.dump(tokenizer.stoi, f)
