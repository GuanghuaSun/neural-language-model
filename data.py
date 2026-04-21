"""
data.py
Loads Word2Vec embeddings and sentence data for the neural language model.
"""

import numpy as np
import torch

EMBEDDING_DIM = 100
CONTEXT_SIZE = 2  # bigram: use 2 previous words as context

SPECIAL_TOKENS = {
    "<START>": 0,
    "</s>": 1,
    "<UNK>": 2,
}


def load_embeddings(vec_file):
    """
    Load pretrained Word2Vec embeddings from vec.txt.
    Returns:
        word_to_id  : dict[str -> int]
        id_to_word  : dict[int -> str]
        embeddings  : np.ndarray of shape (vocab_size, EMBEDDING_DIM)
    """
    word_to_id = dict(SPECIAL_TOKENS)
    id_to_word = {v: k for k, v in SPECIAL_TOKENS.items()}
    rows = []

    # Special token embeddings
    start_emb = np.ones(EMBEDDING_DIM, dtype=np.float32)   # <START>
    end_emb   = np.zeros(EMBEDDING_DIM, dtype=np.float32)  # </s> placeholder
    unk_emb   = np.zeros(EMBEDDING_DIM, dtype=np.float32)  # <UNK>
    rows = [start_emb, end_emb, unk_emb]

    with open(vec_file, encoding="utf-8") as f:
        next(f)  # skip header line
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec  = np.array(parts[1:], dtype=np.float32)
            if len(vec) != EMBEDDING_DIM:
                continue
            word_id = len(word_to_id)
            word_to_id[word] = word_id
            id_to_word[word_id] = word
            rows.append(vec)

    embeddings = np.stack(rows, axis=0)  # (vocab_size, EMBEDDING_DIM)
    return word_to_id, id_to_word, embeddings


def load_sentences(filepath):
    """Load tokenised sentences from a file (one sentence per line)."""
    sentences = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


def sentences_to_examples(sentences, word_to_id, embeddings):
    """
    Convert sentences to (context_vector, target_id) pairs.
    Context = concatenation of CONTEXT_SIZE previous word embeddings.
    Returns:
        X : torch.FloatTensor  (N, EMBEDDING_DIM * CONTEXT_SIZE)
        y : torch.LongTensor   (N,)
    """
    X_list, y_list = [], []
    unk_id = word_to_id["<UNK>"]
    start_id = word_to_id["<START>"]

    for sentence in sentences:
        context_ids = [start_id] * CONTEXT_SIZE
        targets = sentence + ["</s>"]

        for token in targets:
            # Build context vector
            ctx_vecs = [embeddings[cid] for cid in context_ids]
            ctx = np.concatenate(ctx_vecs)  # (EMBEDDING_DIM * CONTEXT_SIZE,)
            X_list.append(ctx)

            # Target word id
            target_id = word_to_id.get(token, unk_id)
            y_list.append(target_id)

            # Slide context window
            context_ids = context_ids[1:] + [target_id]

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y


if __name__ == "__main__":
    print("Loading embeddings...")
    word_to_id, id_to_word, embeddings = load_embeddings("vec.txt")
    print(f"  Vocabulary size : {len(word_to_id):,}")
    print(f"  Embedding matrix: {embeddings.shape}")

    print("\nLoading sentences...")
    train_sents = load_sentences("sentences_train")
    val_sents   = load_sentences("sentences_validation")
    test_sents  = load_sentences("sentences_test")
    print(f"  Train sentences : {len(train_sents):,}")
    print(f"  Val sentences   : {len(val_sents):,}")
    print(f"  Test sentences  : {len(test_sents):,}")

    print("\nBuilding training examples (this may take a minute)...")
    X_train, y_train = sentences_to_examples(train_sents, word_to_id, embeddings)
    print(f"  X_train : {X_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print("\ndata.py OK")
