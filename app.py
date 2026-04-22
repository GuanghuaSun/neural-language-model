"""
app.py
Streamlit demo for the Neural Bigram Language Model.
Run with: streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import math
import json

from model import build_model

EMBEDDING_DIM = 100
CONTEXT_SIZE = 2

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neural Bigram Language Model",
    page_icon="🧠",
    layout="centered",
)

# ── Load model (cached so it only loads once) ─────────────────────────────────

@st.cache_resource
def load_model():
    # Load vocab
    with open("vocab.json") as f:
        data = json.load(f)
    word_to_id = data["word_to_id"]
    id_to_word = {int(k): v for k, v in data["id_to_word"].items()}

    # Load embeddings
    embeddings = np.load("embeddings.npy")

    # Load model
    vocab_size = len(word_to_id)
    model = build_model(vocab_size, hidden_sizes=[100], activation="tanh")
    model.load_state_dict(
        torch.load("best_model.pt", map_location="cpu")
    )
    model.eval()
    return word_to_id, id_to_word, embeddings, model


# ── Inference helpers ─────────────────────────────────────────────────────────

def get_context_vector(context_words, word_to_id, embeddings):
    unk_id = word_to_id["<UNK>"]
    start_id = word_to_id["<START>"]
    context_ids = []
    for w in context_words[-CONTEXT_SIZE:]:
        context_ids.append(word_to_id.get(w.lower(), unk_id))
    while len(context_ids) < CONTEXT_SIZE:
        context_ids.insert(0, start_id)
    vecs = [embeddings[cid] for cid in context_ids]
    return np.concatenate(vecs)


def top_k_predictions(context_vec, model, id_to_word, k=10):
    x = torch.tensor(context_vec, dtype=torch.float32).unsqueeze(0)
    probs = model.predict_proba(x)[0].numpy()
    top_ids = np.argsort(-probs)[:k]
    return [(id_to_word[i], float(probs[i])) for i in top_ids]


def sample_sentence(seed_words, model, word_to_id, id_to_word,
                    embeddings, max_len=20, temperature=1.0):
    unk_id = word_to_id["<UNK>"]
    start_id = word_to_id["<START>"]

    context = []
    for w in seed_words:
        context.append(word_to_id.get(w.lower(), unk_id))
    while len(context) < CONTEXT_SIZE:
        context.insert(0, start_id)
    context = context[-CONTEXT_SIZE:]

    generated = list(seed_words)

    for _ in range(max_len):
        vecs = [embeddings[cid] for cid in context]
        ctx = np.concatenate(vecs)
        x = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)[0]
            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1).numpy()

        next_id = int(np.random.choice(len(probs), p=probs))
        next_word = id_to_word[next_id]

        if next_word == "</s>":
            break

        generated.append(next_word)
        context = context[1:] + [next_id]

    return " ".join(generated)


def sentence_perplexity(sentence, model, word_to_id, id_to_word, embeddings):
    tokens = sentence.strip().lower().split()
    if not tokens:
        return None

    unk_id = word_to_id["<UNK>"]
    start_id = word_to_id["<START>"]
    context_ids = [start_id, start_id]
    targets = tokens + ["</s>"]

    total_nll = 0.0
    for token in targets:
        vecs = [embeddings[cid] for cid in context_ids]
        ctx = np.concatenate(vecs)
        x = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            probs = model.predict_proba(x)[0].numpy()

        target_id = word_to_id.get(token, unk_id)
        p = probs[target_id]
        total_nll += -math.log2(p + 1e-15)
        context_ids = context_ids[1:] + [word_to_id.get(token, unk_id)]

    avg_nll = total_nll / len(targets)
    return 2 ** avg_nll


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    st.title("🧠 Neural Bigram Language Model")
    st.markdown(
        "A PyTorch MLP trained on 10,000 sentences to predict the next word "
        "using Word2Vec embeddings. Trained for 30 epochs · Test PPL: **452.64**"
    )

    word_to_id, id_to_word, embeddings, model = load_model()

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        ["🔮 Next Word Prediction", "✍️ Text Generation", "📊 Sentence Scoring"]
    )

    # ── Tab 1 ──────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Next Word Prediction")
        st.markdown("Enter up to 2 words — the model predicts the most likely next words.")

        col1, col2 = st.columns(2)
        with col1:
            word1 = st.text_input("First word", value="the")
        with col2:
            word2 = st.text_input("Second word", value="bank")

        top_k = st.slider("Show top N predictions", 5, 20, 10)

        if st.button("Predict", key="predict"):
            context = [w for w in [word1.strip(), word2.strip()] if w]
            if not context:
                st.warning("Please enter at least one word.")
            else:
                ctx_vec = get_context_vector(context, word_to_id, embeddings)
                predictions = top_k_predictions(ctx_vec, model, id_to_word, k=top_k)

                st.markdown(f"**Context:** `{' '.join(context)}`")
                st.markdown("**Top predictions:**")

                import pandas as pd
                words = [p[0] for p in predictions]
                probs = [p[1] for p in predictions]
                pct   = [f"{p*100:.2f}%" for p in probs]

                df = pd.DataFrame({"Word": words, "Probability": pct, "Score": probs})
                st.bar_chart(df.set_index("Word")["Score"])
                st.dataframe(df[["Word", "Probability"]], hide_index=True)

    # ── Tab 2 ──────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Text Generation")
        st.markdown("Provide 1–2 seed words and let the model continue the sentence.")

        seed        = st.text_input("Seed words (space-separated)", value="the market")
        max_len     = st.slider("Max new words", 5, 40, 15)
        temperature = st.slider("Temperature (higher = more random)", 0.5, 2.0, 1.0, step=0.1)
        num_samples = st.slider("Number of samples", 1, 5, 3)

        if st.button("Generate", key="generate"):
            seed_words = seed.strip().split()
            if not seed_words:
                st.warning("Please enter at least one seed word.")
            else:
                st.markdown(f"**Seed:** `{seed}`")
                for i in range(num_samples):
                    sentence = sample_sentence(
                        seed_words, model, word_to_id, id_to_word,
                        embeddings, max_len=max_len, temperature=temperature
                    )
                    st.markdown(f"**{i+1}.** {sentence}")

    # ── Tab 3 ──────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Sentence Perplexity Scoring")
        st.markdown(
            "Compare two sentences. Lower perplexity = the model finds it "
            "more natural. Natural word order should score lower than shuffled."
        )

        sent1 = st.text_area("Sentence A", value="the stock market rose sharply today")
        sent2 = st.text_area("Sentence B", value="sharply today market rose stock the")

        if st.button("Score", key="score"):
            ppl1 = sentence_perplexity(sent1, model, word_to_id, id_to_word, embeddings)
            ppl2 = sentence_perplexity(sent2, model, word_to_id, id_to_word, embeddings)

            if ppl1 and ppl2:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentence A — PPL", f"{ppl1:.1f}")
                    st.caption(sent1)
                with col2:
                    st.metric("Sentence B — PPL", f"{ppl2:.1f}")
                    st.caption(sent2)

                if ppl1 < ppl2:
                    st.success(f"✅ Sentence A is {ppl2/ppl1:.1f}x more natural according to the model.")
                elif ppl2 < ppl1:
                    st.success(f"✅ Sentence B is {ppl1/ppl2:.1f}x more natural according to the model.")
                else:
                    st.info("Both sentences have equal perplexity.")

    st.divider()
    st.caption(
        "BigramLM · PyTorch · Word2Vec embeddings (100-dim) · "
        "Vocab: 29,943 · Parameters: 3,044,343 · "
        "Trained 30 epochs · UCSC CSE 142"
    )


if __name__ == "__main__":
    main()
