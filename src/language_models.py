"""
language_models.py
==================
Implements Language Model (LM) based retrieval:
  - Unigram Language Model estimation (MLE)
  - Zero-frequency / OOV problem
  - Jelinek-Mercer (JM) smoothing
  - Dirichlet (Bayesian) smoothing
  - KL-Divergence (KLD) query likelihood scoring
  - Jensen-Shannon Divergence (JSD)
"""

import math
from collections import Counter


# ---------------------------------------------------------------------------
# 1. Maximum Likelihood Estimate (MLE) — unigram document language model
# ---------------------------------------------------------------------------

def mle_lm(doc_tokens: list[str]) -> dict[str, float]:
    """
    MLE unigram language model P(t | D) = count(t, D) / |D|
    """
    n = len(doc_tokens)
    freq = Counter(doc_tokens)
    return {t: c / n for t, c in freq.items()}


def collection_lm(all_tokens: list[list[str]]) -> dict[str, float]:
    """Build collection (background) language model P(t | C)."""
    flat = [t for doc in all_tokens for t in doc]
    return mle_lm(flat)


# ---------------------------------------------------------------------------
# 2. Jelinek-Mercer (JM) Smoothing
# ---------------------------------------------------------------------------

def jm_smoothed_lm(term: str,
                   doc_tokens: list[str],
                   collection_model: dict[str, float],
                   lam: float = 0.5) -> float:
    """
    JM smoothed probability:
        P_JM(t | D) = λ · P(t | D)_MLE  +  (1-λ) · P(t | C)

    Args:
        lam: interpolation weight for document model (0 < λ < 1)
    """
    doc_model = mle_lm(doc_tokens)
    p_doc  = doc_model.get(term, 0.0)
    p_coll = collection_model.get(term, 1e-10)   # avoid zero
    return lam * p_doc + (1 - lam) * p_coll


def jm_query_likelihood(query_terms: list[str],
                        doc_tokens: list[str],
                        collection_model: dict[str, float],
                        lam: float = 0.5) -> float:
    """
    Log query-likelihood score under JM smoothing.
    score = Σ_{t in Q} log P_JM(t | D)
    """
    score = 0.0
    for term in query_terms:
        p = jm_smoothed_lm(term, doc_tokens, collection_model, lam)
        score += math.log(p) if p > 0 else -1e9
    return score


# ---------------------------------------------------------------------------
# 3. Dirichlet (Bayesian) Smoothing
# ---------------------------------------------------------------------------

def dirichlet_smoothed_lm(term: str,
                           doc_tokens: list[str],
                           collection_model: dict[str, float],
                           mu: float = 2000.0) -> float:
    """
    Dirichlet smoothed probability:
        P_Dir(t | D) = (count(t,D) + μ · P(t|C)) / (|D| + μ)

    Args:
        mu: Dirichlet prior strength (typical: 1000-2000)
    """
    doc_len = len(doc_tokens)
    freq    = Counter(doc_tokens)
    tf      = freq.get(term, 0)
    p_coll  = collection_model.get(term, 1e-10)
    return (tf + mu * p_coll) / (doc_len + mu)


def dirichlet_query_likelihood(query_terms: list[str],
                                doc_tokens: list[str],
                                collection_model: dict[str, float],
                                mu: float = 2000.0) -> float:
    """Log query-likelihood under Dirichlet smoothing."""
    score = 0.0
    for term in query_terms:
        p = dirichlet_smoothed_lm(term, doc_tokens, collection_model, mu)
        score += math.log(p) if p > 0 else -1e9
    return score


# ---------------------------------------------------------------------------
# 4. KL-Divergence Retrieval Model
# ---------------------------------------------------------------------------

def kld_score(query_lm: dict[str, float],
              doc_lm: dict[str, float]) -> float:
    """
    KL-Divergence: KLD(Q || D) = Σ P(t|Q) log [P(t|Q) / P(t|D)]
    Lower KLD → higher relevance.
    We return negative KLD so higher = better ranking.
    """
    score = 0.0
    for term, p_q in query_lm.items():
        p_d = doc_lm.get(term, 1e-10)
        score -= p_q * math.log(p_q / p_d)
    return score


# ---------------------------------------------------------------------------
# 5. Jensen-Shannon Divergence (JSD)
# ---------------------------------------------------------------------------

def jsd_score(lm1: dict[str, float], lm2: dict[str, float]) -> float:
    """
    JSD(P || Q) = 0.5 * KLD(P || M) + 0.5 * KLD(Q || M)
    where M = 0.5*(P + Q).
    Returns JSD (0 = identical, 1 = completely different).
    """
    terms = set(lm1) | set(lm2)
    m = {t: 0.5 * (lm1.get(t, 0) + lm2.get(t, 0)) for t in terms}

    def kl(p, q):
        return sum(p.get(t, 0) * math.log(p.get(t, 0) / q[t])
                   for t in terms if p.get(t, 0) > 0 and q[t] > 0)

    return 0.5 * kl(lm1, m) + 0.5 * kl(lm2, m)


# ---------------------------------------------------------------------------
# Retrieval using LM (rank all documents by query likelihood)
# ---------------------------------------------------------------------------

def lm_retrieval(query: str,
                 documents: dict[int, str],
                 method: str = "dirichlet",
                 lam: float = 0.5,
                 mu: float = 2000.0,
                 top_k: int = 5) -> list[tuple[int, float]]:
    """
    Rank documents by query likelihood.

    Args:
        method: 'jm' | 'dirichlet'
    """
    all_tokens = [text.lower().split() for text in documents.values()]
    coll_model = collection_lm(all_tokens)
    query_terms = query.lower().split()

    scores = []
    for did, text in documents.items():
        doc_tokens = text.lower().split()
        if method == "jm":
            score = jm_query_likelihood(query_terms, doc_tokens,
                                        coll_model, lam)
        else:
            score = dirichlet_query_likelihood(query_terms, doc_tokens,
                                               coll_model, mu)
        scores.append((did, score))

    return sorted(scores, key=lambda x: -x[1])[:top_k]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    documents = {
        1: "information retrieval is the science of searching large document collections",
        2: "lucene provides full text search with probabilistic ranking models",
        3: "language models use smoothing to handle zero frequency terms in retrieval",
        4: "dirichlet smoothing is a bayesian approach to language model estimation",
        5: "jelinek mercer smoothing interpolates document and collection language models",
        6: "query likelihood models rank documents by the probability of generating the query",
    }

    query = "language model smoothing retrieval"
    all_tokens = [t.lower().split() for t in documents.values()]
    coll_model = collection_lm(all_tokens)

    print("=== JM Smoothing (λ=0.5) ===")
    for did, score in lm_retrieval(query, documents, method="jm", lam=0.5):
        print(f"  doc {did}: {score:.4f} | {documents[did][:60]}")

    print("\n=== Dirichlet Smoothing (μ=2000) ===")
    for did, score in lm_retrieval(query, documents, method="dirichlet", mu=2000):
        print(f"  doc {did}: {score:.4f} | {documents[did][:60]}")

    # KLD
    q_lm   = mle_lm(query.lower().split())
    d1_lm  = mle_lm(documents[1].lower().split())
    d3_lm  = mle_lm(documents[3].lower().split())
    print(f"\n=== KLD Score ===")
    print(f"  KLD(query || doc1): {kld_score(q_lm, d1_lm):.4f}")
    print(f"  KLD(query || doc3): {kld_score(q_lm, d3_lm):.4f}")

    print(f"\n=== JSD(doc3, doc4) ===")
    print(f"  JSD = {jsd_score(d3_lm, mle_lm(documents[4].lower().split())):.4f}")
