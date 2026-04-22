"""
train.py
Train the PyTorch bigram language model with streaming data loading.
Supports large datasets without loading everything into memory at once.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import time
import math
import random

from data import load_embeddings, load_sentences, EMBEDDING_DIM, CONTEXT_SIZE
from model import build_model


# ── Streaming Dataset ─────────────────────────────────────────────────────────

class StreamingDataset(torch.utils.data.Dataset):
    """
    Builds examples from sentences in chunks to avoid OOM.
    """

    def __init__(self, sentences, word_to_id, embeddings, chunk_size=50000):
        self.sentences  = sentences
        self.word_to_id = word_to_id
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.unk_id     = word_to_id["<UNK>"]
        self.start_id   = word_to_id["<START>"]

        print("  Building examples in chunks...")
        self.X, self.y = self._build_all()

    def _build_all(self):
        X_parts, y_parts = [], []
        chunk = []
        count = 0

        for sent in self.sentences:
            chunk.append(sent)
            count += len(sent) + 1
            if count >= self.chunk_size:
                xc, yc = self._sentences_to_arrays(chunk)
                X_parts.append(xc)
                y_parts.append(yc)
                chunk = []
                count = 0

        if chunk:
            xc, yc = self._sentences_to_arrays(chunk)
            X_parts.append(xc)
            y_parts.append(yc)

        X = torch.from_numpy(np.concatenate(X_parts, axis=0))
        y = torch.from_numpy(np.concatenate(y_parts, axis=0))
        return X, y

    def _sentences_to_arrays(self, sentences):
        X_list, y_list = [], []
        unk_id   = self.unk_id
        start_id = self.start_id
        emb      = self.embeddings
        w2id     = self.word_to_id

        for sentence in sentences:
            context_ids = [start_id] * CONTEXT_SIZE
            targets = sentence + ["</s>"]
            for token in targets:
                ctx = np.concatenate([emb[cid] for cid in context_ids])
                X_list.append(ctx)
                tid = w2id.get(token, unk_id)
                y_list.append(tid)
                context_ids = context_ids[1:] + [tid]

        return (np.array(X_list, dtype=np.float32),
                np.array(y_list,  dtype=np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Perplexity ────────────────────────────────────────────────────────────────

def compute_perplexity(model, X, y, batch_size=4096, device="cpu"):
    model.eval()
    total_nll    = 0.0
    total_tokens = 0
    criterion    = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size].to(device)
            yb = y[start:start + batch_size].to(device)
            nll = criterion(model(xb), yb).item()
            total_nll    += nll
            total_tokens += len(yb)

    return math.exp(total_nll / total_tokens)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading embeddings...")
    word_to_id, id_to_word, embeddings = load_embeddings("vec.txt")
    vocab_size = len(word_to_id)
    print(f"  Vocab size: {vocab_size:,}")

    print("Loading sentences...")
    train_sents = load_sentences(args.train_file)
    val_sents   = load_sentences("sentences_validation")
    test_sents  = load_sentences("sentences_test")
    print(f"  Train: {len(train_sents):,}  Val: {len(val_sents):,}  Test: {len(test_sents):,}")

    print("\nBuilding train dataset...")
    train_ds = StreamingDataset(train_sents, word_to_id, embeddings,
                                chunk_size=args.chunk_size)
    print(f"  Train examples: {len(train_ds):,}")

    print("  Building val/test examples...")
    X_val,  y_val  = train_ds._sentences_to_arrays(val_sents)
    X_test, y_test = train_ds._sentences_to_arrays(test_sents)
    X_val  = torch.from_numpy(X_val);  y_val  = torch.from_numpy(y_val)
    X_test = torch.from_numpy(X_test); y_test = torch.from_numpy(y_test)

    # Build model
    hidden = [int(h) for h in args.hidden.split(",")]
    model  = build_model(vocab_size, hidden_sizes=hidden,
                         activation=args.activation, dropout=args.dropout)
    model  = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: hidden={hidden} activation={args.activation}")
    print(f"Parameters: {total_params:,}")

    # Resume from checkpoint if specified
    if args.resume:
        import os
        if os.path.exists(args.resume):
            model.load_state_dict(torch.load(args.resume, map_location=device))
            print(f"Resumed from checkpoint: {args.resume}")
        else:
            print(f"Warning: checkpoint {args.resume} not found, starting fresh")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    loader    = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda")
    )

    print(f"\nTraining for {args.epochs} epochs...")
    train_ppls, val_ppls = [], []
    best_val_ppl = float("inf")
    start_time   = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches    = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches    += 1

        train_ppl = math.exp(total_loss / batches)
        val_ppl   = compute_perplexity(model, X_val, y_val, device=device)
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_ppl={train_ppl:.1f} | "
              f"val_ppl={val_ppl:.1f} | "
              f"time={time.time()-start_time:.1f}s")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), "best_model.pt")

    # Test evaluation
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    test_ppl = compute_perplexity(model, X_test, y_test, device=device)
    print(f"Test perplexity: {test_ppl:.2f}")

    # Natural vs shuffled
    print("\nNatural vs Shuffled perplexity:")
    shuffled_sents = [random.sample(s, len(s)) for s in test_sents]
    X_shuf, y_shuf = train_ds._sentences_to_arrays(shuffled_sents)
    X_shuf = torch.from_numpy(X_shuf)
    y_shuf = torch.from_numpy(y_shuf)
    shuf_ppl = compute_perplexity(model, X_shuf, y_shuf, device=device)
    print(f"  Natural  PPL: {test_ppl:.2f}")
    print(f"  Shuffled PPL: {shuf_ppl:.2f}")
    print(f"  Shuffled is {shuf_ppl/test_ppl:.1f}x more surprising")

    save_plots(train_ppls, val_ppls, args)
    save_csv(train_ppls, val_ppls, test_ppl)
    print("\nDone. Saved: best_model.pt, training_curves.png, training_log.csv")


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_plots(train_ppls, val_ppls, args):
    epochs = range(1, len(train_ppls) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_ppls, label="Train PPL", color="steelblue")
    axes[0].plot(epochs, val_ppls,   label="Val PPL",   color="tomato")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Perplexity")
    axes[0].set_title("Perplexity over Training")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_ppls, color="tomato", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Perplexity")
    axes[1].set_title(f"Val PPL — hidden={args.hidden} act={args.activation}")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"BigramLM — hidden={args.hidden} act={args.activation} lr={args.lr}")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: training_curves.png")


def save_csv(train_ppls, val_ppls, test_ppl):
    with open("training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_ppl", "val_ppl"])
        for i, (tr, va) in enumerate(zip(train_ppls, val_ppls), 1):
            writer.writerow([i, f"{tr:.4f}", f"{va:.4f}"])
    with open("test_result.txt", "w") as f:
        f.write(f"test_perplexity={test_ppl:.4f}\n")
    print("Saved: training_log.csv, test_result.txt")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=4096)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden",     type=str,   default="100")
    parser.add_argument("--activation", type=str,   default="tanh",
                        choices=["tanh", "relu", "gelu"])
    parser.add_argument("--dropout",    type=float, default=0.0)
    parser.add_argument("--train_file", type=str,   default="sentences_train",
                        help="Training sentences file")
    parser.add_argument("--chunk_size", type=int,   default=50000,
                        help="Examples per chunk when building dataset")
    parser.add_argument("--resume",     type=str,   default="",
                        help="Checkpoint to resume from (e.g. best_model.pt)")
    args = parser.parse_args()
    train(args)
