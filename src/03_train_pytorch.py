#!/usr/bin/env python
# coding: utf-8

# # Aproximación conexionista

# ## Imports y semilla

# In[2]:


from pathlib import Path
import json
import re
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# In[3]:


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# ## Cargar dataset y split

# In[5]:


df = pd.read_csv(Path("../data/processed/dataset.csv"))
df["journal_id"] = df["journal_id"].astype(int)
df["text"] = df["text"].fillna("").astype(str)

X = df["text"].values
y_raw = df["journal_id"].values

# Mapear clases {1,2,3,5} -> {0,1,2,3} para PyTorch
classes = sorted(df["journal_id"].unique().tolist())  # debería ser [1,2,3,5]
class2idx = {c:i for i,c in enumerate(classes)}
idx2class = {i:c for c,i in class2idx.items()}
y = np.array([class2idx[c] for c in y_raw])

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(df)),
    test_size=0.2, random_state=42, stratify=y
)

classes, class2idx


# ## Tokenización simple + vocabulario

# In[6]:


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")  # simple, para texto en inglés

def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())

# Construir vocabulario SOLO con train
from collections import Counter
counter = Counter()
for t in X_train:
    counter.update(tokenize(t))

max_vocab = 30000
min_freq = 2

# Reservas
PAD = "<pad>"
UNK = "<unk>"

vocab = [PAD, UNK]
for tok, freq in counter.most_common():
    if freq < min_freq:
        continue
    vocab.append(tok)
    if len(vocab) >= max_vocab:
        break

stoi = {tok:i for i,tok in enumerate(vocab)}
pad_id = stoi[PAD]
unk_id = stoi[UNK]

len(vocab), vocab[:10]


# In[7]:


def encode(text, max_len=256):
    ids = [stoi.get(tok, unk_id) for tok in tokenize(text)]
    if len(ids) == 0:
        ids = [unk_id]
    return ids[:max_len]

class TextDataset(Dataset):
    def __init__(self, texts, labels, max_len=256):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return encode(self.texts[i], self.max_len), int(self.labels[i])

def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = max(lengths).item()
    x = torch.full((len(seqs), max_len), fill_value=pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return x, lengths, y

train_ds = TextDataset(X_train, y_train, max_len=256)
test_ds  = TextDataset(X_test, y_test, max_len=256)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_dl  = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

next(iter(train_dl))[0].shape


# In[8]:


class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, pad_id):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        self.pad_id = pad_id

    def forward(self, x, lengths):
        # x: [B, T]
        e = self.emb(x)  # [B, T, D]
        mask = (x != self.pad_id).unsqueeze(-1)  # [B, T, 1]
        e = e * mask
        summed = e.sum(dim=1)  # [B, D]
        denom = mask.sum(dim=1).clamp(min=1)  # [B, 1]
        mean = summed / denom
        logits = self.mlp(mean)
        return logits

num_classes = len(classes)
model = MeanPoolClassifier(
    vocab_size=len(vocab),
    emb_dim=128,
    hidden_dim=128,
    num_classes=num_classes,
    pad_id=pad_id
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model


# In[9]:


def run_epoch(model, dl, train=True):
    model.train(train)
    total_loss = 0.0
    all_y, all_pred = [], []

    for x, lengths, yb in dl:
        x, lengths, yb = x.to(device), lengths.to(device), yb.to(device)
        logits = model(x, lengths)
        loss = criterion(logits, yb)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * yb.size(0)
        pred = torch.argmax(logits, dim=1)
        all_y.append(yb.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    avg_loss = total_loss / len(dl.dataset)
    acc = accuracy_score(all_y, all_pred)
    return avg_loss, acc, all_y, all_pred

best_acc = -1.0
best_state = None

for epoch in range(1, 6):  # 5 épocas para empezar
    tr_loss, tr_acc, _, _ = run_epoch(model, train_dl, train=True)
    te_loss, te_acc, y_true, y_pred = run_epoch(model, test_dl, train=False)

    print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | test loss {te_loss:.4f} acc {te_acc:.4f}")

    if te_acc > best_acc:
        best_acc = te_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}


# In[10]:


# Restaurar mejor modelo
model.load_state_dict(best_state)
_, _, y_true, y_pred = run_epoch(model, test_dl, train=False)

# Reporte en ids 0..3
rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print(classification_report(y_true, y_pred, zero_division=0))
print("Confusion matrix:\n", cm)


# In[11]:


OUT_DIR = Path("../reports/pytorch")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("../models/pytorch")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Guardar métricas
payload = {
    "classes_original": classes,            # [1,2,3,5]
    "class2idx": class2idx,
    "best_test_accuracy": float(best_acc),
    "classification_report": rep,
    "confusion_matrix": cm.tolist(),
    "vocab_size": len(vocab),
    "max_len": 256
}
with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

# Guardar modelo + vocab
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "class2idx": class2idx,
        "vocab": vocab,
        "pad_id": pad_id,
        "unk_id": unk_id
    },
    MODELS_DIR / "meanpool_mlp.pt"
)

# Guardar errores en CSV con etiquetas originales (1,2,3,5)
errors = df.iloc[idx_test].copy()
errors["y_true"] = [idx2class[i] for i in y_true]
errors["y_pred"] = [idx2class[i] for i in y_pred]
errors = errors[errors["y_true"] != errors["y_pred"]]
errors.to_csv(OUT_DIR / "errors.csv", index=False)

OUT_DIR / "metrics.json", OUT_DIR / "errors.csv"

