"""
advanced_ir.py
==============
Advanced IR techniques:
  - LSI (Latent Semantic Indexing) via Truncated SVD
  - Simple Word2Vec-style co-occurrence embeddings
  - Rocchio Relevance Feedback
  - RLM (Relevance Language Model) feedback
  - BERT-style reranking stub (demonstrates API usage)
"""

import math
import numpy as np
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# 1. LSI — Latent Semantic Indexing
# ---------------------------------------------------------------------------

def build_term_doc_matrix(documents: dict[int, str]) -> tuple:
    """
    Build a raw TF term-document matrix.

    Returns:
        matrix (ndarray), terms (list), doc_ids (list)
    """
    doc_ids = list(documents.keys())
    all_tokens = [text.lower().split() for text in documents.values()]
    terms = sorted(set(t for tokens in all_tokens for t in tokens))
    term_idx = {t: i for i, t in enumerate(terms)}

    matrix = np.zeros((len(terms), len(doc_ids)), dtype=float)
    for j, tokens in enumerate(all_tokens):
        for t in tokens:
            matrix[term_idx[t], j] += 1
    return matrix, terms, doc_ids


def lsi(documents: dict[int, str], k: int = 2) -> dict:
    """
    Latent Semantic Indexing via rank-k SVD.
    Projects documents and terms into a k-dimensional latent space.

    Returns dict with:
      - 'U'       : term-concept matrix (|terms| × k)
      - 'S'       : singular values (k,)
      - 'Vt'      : doc-concept matrix (k × |docs|)
      - 'terms'   : list of terms
      - 'doc_ids' : list of doc IDs
    """
    A, terms, doc_ids = build_term_doc_matrix(documents)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # Truncate to rank k
    U_k  = U[:, :k]
    S_k  = S[:k]
    Vt_k = Vt[:k, :]
    return {"U": U_k, "S": S_k, "Vt": Vt_k,
            "terms": terms, "doc_ids": doc_ids}


def lsi_query(query: str, lsi_model: dict, top_k: int = 3) -> list:
    """
    Fold a query into LSI space and rank documents by cosine similarity.
    """
    terms    = lsi_model["terms"]
    term_idx = {t: i for i, t in enumerate(terms)}
    U_k, S_k, Vt_k = lsi_model["U"], lsi_model["S"], lsi_model["Vt"]

    # Build query vector in original space
    q_vec = np.zeros(len(terms))
    for t in query.lower().split():
        if t in term_idx:
            q_vec[term_idx[t]] += 1

    # Project query: q_lsi = q^T · U_k · S_k^{-1}  (pseudo-inverse fold-in)
    q_lsi = q_vec @ U_k / S_k            # shape: (k,)

    # Document vectors in LSI space: columns of S_k * Vt_k
    doc_vecs = (np.diag(S_k) @ Vt_k).T   # shape: (n_docs, k)

    # Cosine similarity
    def cosine(a, b):
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / n) if n > 0 else 0.0

    scores = [(lsi_model["doc_ids"][j], cosine(q_lsi, doc_vecs[j]))
              for j in range(len(lsi_model["doc_ids"]))]
    return sorted(scores, key=lambda x: -x[1])[:top_k]


# ---------------------------------------------------------------------------
# 2. Co-occurrence Word Embeddings (simplified Word2Vec-style)
# ---------------------------------------------------------------------------

def build_cooccurrence_matrix(texts: list[str],
                               window: int = 2) -> tuple:
    """
    Build a symmetric co-occurrence matrix with a sliding window.

    Returns:
        matrix (ndarray), vocab (list of terms)
    """
    vocab = sorted(set(t for text in texts for t in text.lower().split()))
    w2i   = {w: i for i, w in enumerate(vocab)}
    n     = len(vocab)
    co    = np.zeros((n, n), dtype=float)

    for text in texts:
        tokens = text.lower().split()
        for i, t in enumerate(tokens):
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if i != j:
                    co[w2i[t], w2i[tokens[j]]] += 1
    return co, vocab


def ppmi(co_matrix: np.ndarray) -> np.ndarray:
    """
    Positive Pointwise Mutual Information (PPMI) weighting.
    PPMI(w, c) = max(0, log2(P(w,c) / P(w)P(c)))
    """
    total  = co_matrix.sum()
    p_wc   = co_matrix / total
    p_w    = p_wc.sum(axis=1, keepdims=True)
    p_c    = p_wc.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log2(np.where(p_w * p_c > 0, p_wc / (p_w * p_c), 1e-10))
    return np.maximum(pmi, 0)


# ---------------------------------------------------------------------------
# 3. Rocchio Relevance Feedback
# ---------------------------------------------------------------------------

def rocchio(query_vec: np.ndarray,
            relevant_vecs: list[np.ndarray],
            non_relevant_vecs: list[np.ndarray],
            alpha: float = 1.0,
            beta: float = 0.75,
            gamma: float = 0.25) -> np.ndarray:
    """
    Rocchio algorithm for query modification:
        q' = α·q + β·(1/|R|)Σ d_r − γ·(1/|NR|)Σ d_nr

    Args:
        alpha, beta, gamma: weights for original query, relevant, non-relevant
    """
    rel_centroid    = (np.mean(relevant_vecs, axis=0)
                       if relevant_vecs else np.zeros_like(query_vec))
    nonrel_centroid = (np.mean(non_relevant_vecs, axis=0)
                       if non_relevant_vecs else np.zeros_like(query_vec))

    new_query = alpha * query_vec + beta * rel_centroid - gamma * nonrel_centroid
    return np.maximum(new_query, 0)   # clip negatives (standard practice)


# ---------------------------------------------------------------------------
# 4. BERT Reranking Stub
# ---------------------------------------------------------------------------

def bert_rerank(query: str,
                candidates: list[tuple[int, str]],
                top_k: int = 5) -> list[tuple[int, float]]:
    """
    Stub for BERT-based cross-encoder reranking.
    In production, replace the similarity function with a
    HuggingFace cross-encoder:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        scores = model.predict([(query, text) for _, text in candidates])

    Here we simulate with a simple token-overlap score.
    """
    q_tokens = set(query.lower().split())
    scores = []
    for doc_id, text in candidates:
        d_tokens = set(text.lower().split())
        # Simulated relevance: Jaccard overlap
        sim = len(q_tokens & d_tokens) / len(q_tokens | d_tokens) if q_tokens | d_tokens else 0
        scores.append((doc_id, round(sim, 4)))
    return sorted(scores, key=lambda x: -x[1])[:top_k]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    documents = {
        1: "information retrieval search index lucene",
        2: "lucene full text search engine ranking",
        3: "language model probability smoothing retrieval",
        4: "bm25 probabilistic ranking information retrieval",
        5: "word embeddings semantic similarity vector space",
    }

    # --- LSI ---
    print("=== LSI (k=2) ===")
    lsi_model = lsi(documents, k=2)
    results = lsi_query("retrieval ranking", lsi_model)
    for did, score in results:
        print(f"  doc {did}: {score:.4f} | {documents[did]}")

    # --- Co-occurrence PPMI ---
    texts = list(documents.values())
    co, vocab = build_cooccurrence_matrix(texts, window=2)
    ppmi_mat  = ppmi(co)
    print(f"\n=== PPMI Matrix ===")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  PPMI shape: {ppmi_mat.shape}")
    # Show top co-occurrence for 'retrieval'
    if "retrieval" in vocab:
        idx   = vocab.index("retrieval")
        top_i = np.argsort(-ppmi_mat[idx])[:5]
        print(f"  Top words co-occurring with 'retrieval':")
        for i in top_i:
            print(f"    {vocab[i]}: PPMI={ppmi_mat[idx, i]:.3f}")

    # --- Rocchio ---
    dim = 5
    q      = np.array([1.0, 0.5, 0.0, 0.2, 0.0])
    r_docs = [np.array([0.9, 0.8, 0.1, 0.3, 0.0]),
              np.array([1.0, 0.6, 0.0, 0.4, 0.1])]
    nr_docs = [np.array([0.1, 0.0, 0.9, 0.1, 0.8])]
    q_new = rocchio(q, r_docs, nr_docs)
    print(f"\n=== Rocchio Feedback ===")
    print(f"  Original query vec : {q}")
    print(f"  Expanded query vec : {q_new.round(4)}")

    # --- BERT Reranking ---
    candidates = [(did, text) for did, text in documents.items()]
    print(f"\n=== BERT Reranking (simulated) ===")
    for did, score in bert_rerank("retrieval ranking model", candidates):
        print(f"  doc {did}: sim={score} | {documents[did]}")
