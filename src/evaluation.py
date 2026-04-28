"""
evaluation.py
=============
Standard IR evaluation metrics:
  - Precision, Recall, F-measure
  - Precision@K, R-Precision
  - Average Precision (AP) & MAP
  - GMAP (Geometric MAP)
  - MRR (Mean Reciprocal Rank)
  - nDCG (Normalized Discounted Cumulative Gain)
  - Kappa measure (inter-annotator agreement)
"""

import math
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. Set-based metrics
# ---------------------------------------------------------------------------

def precision(retrieved: list, relevant: set) -> float:
    """P = |retrieved ∩ relevant| / |retrieved|"""
    if not retrieved:
        return 0.0
    return len(set(retrieved) & relevant) / len(retrieved)


def recall(retrieved: list, relevant: set) -> float:
    """R = |retrieved ∩ relevant| / |relevant|"""
    if not relevant:
        return 0.0
    return len(set(retrieved) & relevant) / len(relevant)


def f_measure(retrieved: list, relevant: set, beta: float = 1.0) -> float:
    """
    F_β = (1 + β²) · P · R / (β²·P + R)
    F1 when β=1 (equal weight); β<1 favours precision, β>1 recall.
    """
    p = precision(retrieved, relevant)
    r = recall(retrieved, relevant)
    if p + r == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


# ---------------------------------------------------------------------------
# 2. Precision@K and R-Precision
# ---------------------------------------------------------------------------

def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    """P@K — precision of the top-K retrieved documents."""
    return precision(ranked[:k], relevant)


def r_precision(ranked: list, relevant: set) -> float:
    """R-Prec — P@R where R = |relevant|."""
    r = len(relevant)
    return precision(ranked[:r], relevant)


# ---------------------------------------------------------------------------
# 3. Average Precision (AP) & MAP
# ---------------------------------------------------------------------------

def average_precision(ranked: list, relevant: set) -> float:
    """
    AP = (1 / |R|) Σ_{k: doc_k is relevant} P@k
    Rewards systems that rank relevant documents high.
    """
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for k, doc in enumerate(ranked, start=1):
        if doc in relevant:
            hits += 1
            total += hits / k
    return total / len(relevant)


def mean_average_precision(queries_results: dict) -> float:
    """
    MAP = mean of AP over all queries.

    Args:
        queries_results: {query_id: {'ranked': [...], 'relevant': set(...)}}
    """
    aps = [average_precision(v['ranked'], v['relevant'])
           for v in queries_results.values()]
    return sum(aps) / len(aps) if aps else 0.0


def gmap(queries_results: dict, epsilon: float = 1e-3) -> float:
    """
    GMAP = exp(mean(log(AP + ε)))
    More sensitive to poorly performing queries than MAP.
    """
    aps = [average_precision(v['ranked'], v['relevant']) + epsilon
           for v in queries_results.values()]
    return math.exp(sum(math.log(ap) for ap in aps) / len(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# 4. MRR — Mean Reciprocal Rank
# ---------------------------------------------------------------------------

def reciprocal_rank(ranked: list, relevant: set) -> float:
    """RR = 1 / rank_of_first_relevant_doc (0 if none found)."""
    for k, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / k
    return 0.0


def mean_reciprocal_rank(queries_results: dict) -> float:
    """MRR = mean RR over all queries."""
    rrs = [reciprocal_rank(v['ranked'], v['relevant'])
           for v in queries_results.values()]
    return sum(rrs) / len(rrs) if rrs else 0.0


# ---------------------------------------------------------------------------
# 5. nDCG — Normalized Discounted Cumulative Gain
# ---------------------------------------------------------------------------

def dcg(gains: list[float], k: int) -> float:
    """DCG@K = Σ_{i=1}^{K} gain_i / log2(i+1)"""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg(ranked: list, relevance_grades: dict, k: int) -> float:
    """
    nDCG@K = DCG@K / IDCG@K

    Args:
        ranked           : ordered list of retrieved doc IDs
        relevance_grades : {doc_id: relevance_grade}  (0, 1, 2, 3 …)
        k                : cutoff rank
    """
    gains = [relevance_grades.get(doc, 0) for doc in ranked]
    ideal_gains = sorted(relevance_grades.values(), reverse=True)
    actual_dcg  = dcg(gains, k)
    ideal_dcg   = dcg(ideal_gains, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------------------------------------------------------------------------
# 6. Kappa Measure (inter-annotator agreement)
# ---------------------------------------------------------------------------

def kappa(agreed: int, total: int,
          p_e_rel: float, p_e_irr: float) -> float:
    """
    Cohen's Kappa:  κ = (P_o - P_e) / (1 - P_e)

    Args:
        agreed    : number of documents annotators agreed on
        total     : total documents judged
        p_e_rel   : probability both annotators independently judge relevant
        p_e_irr   : probability both independently judge non-relevant
    """
    p_o = agreed / total
    p_e = p_e_rel + p_e_irr
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 0.0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Single-query example
    relevant = {1, 3, 5, 7, 9}
    ranked   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print("=== Set-Based Metrics ===")
    print(f"  Precision     : {precision(ranked, relevant):.4f}")
    print(f"  Recall        : {recall(ranked, relevant):.4f}")
    print(f"  F1            : {f_measure(ranked, relevant, beta=1):.4f}")
    print(f"  P@5           : {precision_at_k(ranked, relevant, 5):.4f}")
    print(f"  R-Precision   : {r_precision(ranked, relevant):.4f}")

    print("\n=== Ranking-Based Metrics ===")
    print(f"  AP            : {average_precision(ranked, relevant):.4f}")
    print(f"  RR            : {reciprocal_rank(ranked, relevant):.4f}")

    # Multi-query MAP / GMAP / MRR
    queries = {
        "q1": {"ranked": [1, 2, 3, 4, 5],  "relevant": {1, 3}},
        "q2": {"ranked": [3, 1, 5, 2, 4],  "relevant": {1, 2, 5}},
        "q3": {"ranked": [2, 4, 6, 8, 10], "relevant": {1, 3, 5}},
    }
    print(f"\n=== Multi-Query Metrics ===")
    print(f"  MAP           : {mean_average_precision(queries):.4f}")
    print(f"  GMAP          : {gmap(queries):.4f}")
    print(f"  MRR           : {mean_reciprocal_rank(queries):.4f}")

    # nDCG
    rel_grades = {1: 3, 2: 1, 3: 2, 4: 0, 5: 3, 6: 1}
    ranked_docs = [1, 3, 2, 5, 6, 4]
    print(f"\n=== nDCG ===")
    for k in [3, 5, 6]:
        print(f"  nDCG@{k}        : {ndcg(ranked_docs, rel_grades, k):.4f}")

    # Kappa
    kap = kappa(agreed=80, total=100, p_e_rel=0.25, p_e_irr=0.45)
    print(f"\n=== Kappa ===")
    print(f"  κ = {kap:.4f}  ({'Fair' if kap < 0.6 else 'Substantial'})")
