import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import re
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# --- Step 1: Sample Raw Paragraph ---
text = """I want to build a next word predictor. It should take a sentence and suggest the next word. This helps in learning how LSTM works in NLP tasks."""

# --- Step 2: Tokenization ---
def tokenize(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()

tokens = tokenize(text)

# --- Step 3: Vocabulary ---
counter = Counter(tokens)
vocab = {word: idx for idx, (word, _) in enumerate(counter.items())}
inv_vocab = {idx: word for word, idx in vocab.items()}
token_ids = [vocab[word] for word in tokens]
vocab_size = len(vocab)

# --- Step 4: Context → Next Word Pairs ---
context_size = 4
inputs, targets = [], []
for i in range(context_size, len(token_ids)):
    context = token_ids[i - context_size:i]
    target = token_ids[i]
    inputs.append(torch.tensor(context))
    targets.append(torch.tensor(target))

# --- Step 5: Dataset & Dataloader ---
class NextWordDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

dataset = NextWordDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# --- Step 6: Model Definition ---
class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # [B, T] -> [B, T, E]
        out, (h_n, _) = self.lstm(x)  # [B, T, H], h_n: [1, B, H]
        return self.fc(h_n[-1])  # [B, H] -> [B, V]

# --- Step 7: Train the Model ---
embed_size = 50
hidden_size = 128
model = NextWordLSTM(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []
epochs = 30
for epoch in range(epochs):
    total_loss = 0
    for X, y in loader:
        optimizer.zero_grad()
        logits = model(X)              # [B, V]
        loss = criterion(logits, y)    # compare with next word IDs
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# --- Step 8: Inference Function ---
def predict_next_word(model, input_words):
    model.eval()
    with torch.no_grad():
        ids = [vocab[word] for word in input_words[-context_size:]]
        x = torch.tensor(ids).unsqueeze(0)  # [1, context_size]
        logits = model(x)  # [1, vocab_size]
        pred_id = torch.argmax(logits, dim=-1).item()
        return inv_vocab[pred_id]

# --- Step 9: Predict a Next Word ---
test_input = ["how", "lstm", "works", "in"]
predicted = predict_next_word(model, test_input)
print("Input:", test_input)
print("Predicted next word:", predicted)

# --- Step 10: Plot Training Loss ---
plt.plot(losses, marker='o')
plt.title("Training Loss vs Epoch (Next Word Prediction)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
