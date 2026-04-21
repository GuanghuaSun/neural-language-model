"""
model.py
PyTorch MLP bigram language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLM(nn.Module):
    """
    A simple MLP language model.

    Architecture:
        Input  : concatenation of 2 Word2Vec embeddings  (200-dim)
        Hidden : one or two fully-connected layers + activation
        Output : logits over full vocabulary  (vocab_size)

    Training objective: cross-entropy (next-word prediction).
    """

    def __init__(self, input_dim, hidden_sizes, vocab_size, activation="tanh", dropout=0.0):
        """
        Args:
            input_dim    : EMBEDDING_DIM * CONTEXT_SIZE  (200)
            hidden_sizes : list of ints, e.g. [100] or [200, 100]
            vocab_size   : number of output classes
            activation   : "tanh" | "relu" | "gelu"
            dropout      : dropout probability (0 = disabled)
        """
        super().__init__()

        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (batch, input_dim) -> logits: (batch, vocab_size)"""
        return self.net(x)

    def predict_proba(self, x):
        """Return probability distribution over vocabulary."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


def build_model(vocab_size, hidden_sizes=(100,), activation="tanh", dropout=0.0):
    """Convenience factory used by train.py and app.py."""
    INPUT_DIM = 200  # 2 * EMBEDDING_DIM
    return BigramLM(INPUT_DIM, list(hidden_sizes), vocab_size, activation, dropout)


if __name__ == "__main__":
    VOCAB = 29944
    model = build_model(VOCAB, hidden_sizes=[100], activation="tanh")
    print(model)

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")

    # Quick forward pass
    x = torch.randn(4, 200)
    out = model(x)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("model.py OK")
