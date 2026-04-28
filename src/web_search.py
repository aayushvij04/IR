"""
web_search.py
=============
Implements web search algorithms:
  - Shingling & Jaccard Similarity (near-duplicate detection)
  - PageRank (Random Surfer's Algorithm)
  - HITS (Hyperlink-Induced Topic Search) — hub & authority scores
"""

import math
import hashlib
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. Shingling & Jaccard Similarity
# ---------------------------------------------------------------------------

def shingle(text: str, k: int = 3) -> set[str]:
    """
    Generate k-shingles (character-level) from text.
    A k-shingle is a contiguous sequence of k characters.
    """
    text = text.lower().replace(" ", "")
    return {text[i: i + k] for i in range(len(text) - k + 1)}


def word_shingle(text: str, k: int = 2) -> set[str]:
    """Generate k-word shingles (token-level)."""
    tokens = text.lower().split()
    return {" ".join(tokens[i: i + k]) for i in range(len(tokens) - k + 1)}


def jaccard_similarity(s1: set, s2: set) -> float:
    """Jaccard(A, B) = |A ∩ B| / |A ∪ B|"""
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def near_duplicate_pairs(documents: dict[int, str],
                          k: int = 3,
                          threshold: float = 0.5) -> list[tuple]:
    """
    Find near-duplicate document pairs with Jaccard ≥ threshold.
    Returns [(doc_i, doc_j, similarity), ...]
    """
    shingles = {did: shingle(text, k) for did, text in documents.items()}
    doc_ids  = list(documents.keys())
    pairs    = []
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            d1, d2 = doc_ids[i], doc_ids[j]
            sim = jaccard_similarity(shingles[d1], shingles[d2])
            if sim >= threshold:
                pairs.append((d1, d2, round(sim, 4)))
    return pairs


# ---------------------------------------------------------------------------
# 2. PageRank — Random Surfer's Algorithm
# ---------------------------------------------------------------------------

def pagerank(graph: dict[int, list[int]],
             d: float = 0.85,
             max_iter: int = 100,
             tol: float = 1e-6) -> dict[int, float]:
    """
    Iterative PageRank.

    Args:
        graph   : {node: [list of nodes it links TO]}
        d       : damping factor (default 0.85)
        max_iter: max iterations
        tol     : convergence tolerance

    Returns:
        {node: pagerank_score}
    """
    nodes = set(graph.keys()) | {n for nbrs in graph.values() for n in nbrs}
    N     = len(nodes)

    # Initialise uniform
    pr = {n: 1.0 / N for n in nodes}

    # Build in-link map
    in_links: dict = defaultdict(list)
    for src, targets in graph.items():
        for tgt in targets:
            in_links[tgt].append(src)

    # Out-degree (handle dangling nodes)
    out_degree = {n: len(graph.get(n, [])) for n in nodes}

    for iteration in range(max_iter):
        new_pr = {}
        for n in nodes:
            rank_sum = sum(
                pr[m] / out_degree[m]
                for m in in_links[n]
                if out_degree[m] > 0
            )
            new_pr[n] = (1 - d) / N + d * rank_sum

        # Check convergence
        diff = sum(abs(new_pr[n] - pr[n]) for n in nodes)
        pr = new_pr
        if diff < tol:
            print(f"  PageRank converged in {iteration + 1} iterations")
            break

    # Normalize
    total = sum(pr.values())
    return {n: v / total for n, v in pr.items()}


# ---------------------------------------------------------------------------
# 3. HITS — Hub & Authority Scores
# ---------------------------------------------------------------------------

def hits(graph: dict[int, list[int]],
         max_iter: int = 100,
         tol: float = 1e-6) -> tuple[dict, dict]:
    """
    HITS algorithm.

    Args:
        graph: {node: [list of nodes it links TO]}

    Returns:
        (authority_scores, hub_scores) — both normalized
    """
    nodes = set(graph.keys()) | {n for nbrs in graph.values() for n in nbrs}

    hub   = {n: 1.0 for n in nodes}
    auth  = {n: 1.0 for n in nodes}

    # Build in-link map
    in_links: dict = defaultdict(list)
    for src, targets in graph.items():
        for tgt in targets:
            in_links[tgt].append(src)

    for iteration in range(max_iter):
        # Update authority: a(n) = Σ h(m) for all m→n
        new_auth = {n: sum(hub[m] for m in in_links.get(n, []))
                    for n in nodes}
        # Update hub: h(n) = Σ a(m) for all n→m
        new_hub  = {n: sum(new_auth.get(m, 0) for m in graph.get(n, []))
                    for n in nodes}

        # Normalize
        auth_norm = math.sqrt(sum(v ** 2 for v in new_auth.values())) or 1.0
        hub_norm  = math.sqrt(sum(v ** 2 for v in new_hub.values()))  or 1.0
        new_auth = {n: v / auth_norm for n, v in new_auth.items()}
        new_hub  = {n: v / hub_norm  for n, v in new_hub.items()}

        # Convergence
        diff = (sum(abs(new_auth[n] - auth[n]) for n in nodes) +
                sum(abs(new_hub[n]  - hub[n])  for n in nodes))
        auth, hub = new_auth, new_hub
        if diff < tol:
            print(f"  HITS converged in {iteration + 1} iterations")
            break

    return auth, hub


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Shingling ---
    docs = {
        1: "information retrieval is the activity of obtaining information resources",
        2: "information retrieval is obtaining information from resources",   # near-dup
        3: "lucene provides full text search with ranking and indexing",
        4: "lucene search engine supports full text ranking and retrieval",
    }
    print("=== Near-Duplicate Detection (Jaccard ≥ 0.3, k=3 shingles) ===")
    for d1, d2, sim in near_duplicate_pairs(docs, k=3, threshold=0.3):
        print(f"  doc {d1} vs doc {d2}: Jaccard = {sim}")

    # --- PageRank ---
    web_graph = {
        1: [2, 3],
        2: [3],
        3: [1],
        4: [3],
        5: [4, 1],
    }
    print("\n=== PageRank ===")
    pr = pagerank(web_graph)
    for node, score in sorted(pr.items(), key=lambda x: -x[1]):
        print(f"  Page {node}: PR = {score:.4f}")

    # --- HITS ---
    print("\n=== HITS ===")
    auth, hub = hits(web_graph)
    print(f"  {'Node':>5}  {'Authority':>10}  {'Hub':>10}")
    for n in sorted(auth):
        print(f"  {n:>5}  {auth[n]:>10.4f}  {hub[n]:>10.4f}")
