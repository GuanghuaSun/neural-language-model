"""
train.py
Train the PyTorch bigram language model and evaluate with perplexity.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import time
import math

from data import load_embeddings, load_sentences, sentences_to_examples
from model import build_model


# ── Perplexity ────────────────────────────────────────────────────────────────

def compute_perplexity(model, X, y, batch_size=4096, device="cpu"):
    """
    Compute perplexity on a dataset.
    PPL = exp(average negative log-likelihood)
    Lower is better.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size].to(device)
            yb = y[start:start + batch_size].to(device)
            logits = model(xb)
            nll = criterion(logits, yb).item()
            total_nll += nll
            total_tokens += len(yb)

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return ppl


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    word_to_id, id_to_word, embeddings = load_embeddings("vec.txt")
    vocab_size = len(word_to_id)
    print(f"  Vocab size: {vocab_size:,}")

    train_sents = load_sentences("sentences_train")
    val_sents   = load_sentences("sentences_validation")
    test_sents  = load_sentences("sentences_test")

    print("  Building examples...")
    X_train, y_train = sentences_to_examples(train_sents, word_to_id, embeddings)
    X_val,   y_val   = sentences_to_examples(val_sents,   word_to_id, embeddings)
    X_test,  y_test  = sentences_to_examples(test_sents,  word_to_id, embeddings)
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # Build model
    hidden = [int(h) for h in args.hidden.split(",")]
    model = build_model(vocab_size, hidden_sizes=hidden,
                        activation=args.activation, dropout=args.dropout)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: hidden={hidden} activation={args.activation} "
          f"dropout={args.dropout}")
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training
    print(f"\nTraining for {args.epochs} epochs, batch_size={args.batch_size}...")
    train_ppls, val_ppls = [], []
    best_val_ppl = float("inf")

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Shuffle
        perm = torch.randperm(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        total_loss = 0.0
        batches = 0
        for start in range(0, len(X_train), args.batch_size):
            xb = X_train[start:start + args.batch_size]
            yb = y_train[start:start + args.batch_size]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        train_ppl = math.exp(avg_loss)

        # Validation perplexity
        val_ppl = compute_perplexity(model, X_val, y_val, device=device)

        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={avg_loss:.4f} | "
              f"train_ppl={train_ppl:.1f} | "
              f"val_ppl={val_ppl:.1f} | "
              f"time={elapsed:.1f}s")

        # Save best model
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), "best_model.pt")

    # Test perplexity
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    test_ppl = compute_perplexity(model, X_test, y_test, device=device)
    print(f"Test perplexity: {test_ppl:.2f}")

    # Save training curves
    save_plots(train_ppls, val_ppls, args)
    save_csv(train_ppls, val_ppls, test_ppl)

    # Validate: natural vs shuffled sentences
    print("\nNatural vs Shuffled perplexity:")
    nat_ppl  = compute_perplexity(model, X_test, y_test, device=device)

    import random
    shuffled_sents = []
    for s in test_sents:
        s2 = s[:]
        random.shuffle(s2)
        shuffled_sents.append(s2)
    X_shuf, y_shuf = sentences_to_examples(shuffled_sents, word_to_id, embeddings)
    shuf_ppl = compute_perplexity(model, X_shuf, y_shuf, device=device)

    print(f"  Natural  PPL: {nat_ppl:.2f}")
    print(f"  Shuffled PPL: {shuf_ppl:.2f}")
    print(f"  Shuffled is {shuf_ppl/nat_ppl:.1f}x more surprising")

    print("\nDone. Saved: best_model.pt, training_curves.png, training_log.csv")
    return word_to_id, id_to_word, embeddings, model


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_plots(train_ppls, val_ppls, args):
    epochs = range(1, len(train_ppls) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_ppls, label="Train PPL", color="steelblue")
    axes[0].plot(epochs, val_ppls,   label="Val PPL",   color="tomato")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Perplexity")
    axes[0].set_title("Perplexity over Training")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_ppls, color="tomato", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Perplexity")
    axes[1].set_title(f"Val PPL — hidden={args.hidden} act={args.activation}")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"BigramLM Training — hidden={args.hidden} "
                 f"act={args.activation} lr={args.lr}", fontsize=11)
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
    parser = argparse.ArgumentParser(description="Train BigramLM")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=4096)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden",     type=str,   default="100",
                        help="Hidden layer sizes, comma-separated e.g. '200,100'")
    parser.add_argument("--activation", type=str,   default="tanh",
                        choices=["tanh", "relu", "gelu"])
    parser.add_argument("--dropout",    type=float, default=0.0)
    args = parser.parse_args()

    train(args)
