"""
boolean_retrieval.py
====================
Implements core Week-1 IR concepts:
  - Term-Document Incidence Matrix
  - Inverted Index construction
  - Boolean query processing (AND, OR, NOT)
  - Skip pointers for faster AND merging
"""

import math
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. Term-Document Incidence Matrix
# ---------------------------------------------------------------------------

def build_incidence_matrix(documents: dict[str, str]) -> dict:
    """
    Build a binary term-document incidence matrix.

    Args:
        documents: {doc_id: text}

    Returns:
        matrix: {term: {doc_id: 0|1}}
    """
    terms = set()
    tokenized = {}
    for doc_id, text in documents.items():
        tokens = text.lower().split()
        tokenized[doc_id] = set(tokens)
        terms.update(tokens)

    matrix = {}
    for term in terms:
        matrix[term] = {doc_id: int(term in tokenized[doc_id])
                        for doc_id in documents}
    return matrix


def boolean_and_matrix(matrix: dict, term1: str, term2: str,
                        doc_ids: list) -> list:
    """Return doc_ids where both term1 AND term2 appear."""
    row1 = matrix.get(term1, {})
    row2 = matrix.get(term2, {})
    return [d for d in doc_ids if row1.get(d, 0) and row2.get(d, 0)]


# ---------------------------------------------------------------------------
# 2. Inverted Index
# ---------------------------------------------------------------------------

class InvertedIndex:
    """
    Builds and stores an inverted index from a collection of documents.
    Supports Boolean retrieval and postings-list operations.
    """

    def __init__(self):
        self.index: dict[str, list[int]] = defaultdict(list)
        self.doc_freq: dict[str, int] = {}
        self.num_docs = 0

    def build(self, documents: dict[int, str]):
        """
        Construct the inverted index.

        Args:
            documents: {doc_id (int): text}
        """
        self.num_docs = len(documents)
        for doc_id, text in documents.items():
            tokens = set(text.lower().split())   # unique terms per doc
            for token in tokens:
                self.index[token].append(doc_id)

        # Sort postings and compute document frequencies
        for term in self.index:
            self.index[term].sort()
            self.doc_freq[term] = len(self.index[term])

    def get_postings(self, term: str) -> list[int]:
        return sorted(self.index.get(term.lower(), []))

    # ----------------------------------------------------------------
    # Boolean operations on sorted postings lists
    # ----------------------------------------------------------------

    def AND(self, term1: str, term2: str) -> list[int]:
        """Merge two postings lists (intersection). O(p1 + p2)."""
        p1 = self.get_postings(term1)
        p2 = self.get_postings(term2)
        result, i, j = [], 0, 0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                result.append(p1[i]); i += 1; j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j += 1
        return result

    def OR(self, term1: str, term2: str) -> list[int]:
        """Union of two postings lists."""
        p1 = self.get_postings(term1)
        p2 = self.get_postings(term2)
        result, i, j = [], 0, 0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                result.append(p1[i]); i += 1; j += 1
            elif p1[i] < p2[j]:
                result.append(p1[i]); i += 1
            else:
                result.append(p2[j]); j += 1
        result.extend(p1[i:]); result.extend(p2[j:])
        return result

    def NOT(self, term: str) -> list[int]:
        """Complement of a postings list w.r.t. all doc IDs."""
        postings = set(self.get_postings(term))
        return [d for d in range(self.num_docs) if d not in postings]

    def AND_NOT(self, term1: str, term2: str) -> list[int]:
        """Documents containing term1 but NOT term2."""
        p1 = self.get_postings(term1)
        p2 = set(self.get_postings(term2))
        return [d for d in p1 if d not in p2]


# ---------------------------------------------------------------------------
# 3. Skip Pointers for faster AND merging
# ---------------------------------------------------------------------------

def add_skip_pointers(postings: list[int]) -> list:
    """
    Augment a sorted postings list with skip pointers.
    Skip step = floor(sqrt(len(postings))).

    Returns list of (doc_id, skip_target_index | None).
    """
    n = len(postings)
    step = max(1, int(math.sqrt(n)))
    result = []
    for i, doc_id in enumerate(postings):
        skip = i + step if (i % step == 0 and i + step < n) else None
        result.append((doc_id, skip))
    return result


def and_with_skips(p1_skips: list, p2_skips: list) -> list[int]:
    """
    AND merge using skip pointers.
    Each element: (doc_id, skip_index | None).
    """
    result = []
    i, j = 0, 0
    while i < len(p1_skips) and j < len(p2_skips):
        d1, s1 = p1_skips[i]
        d2, s2 = p2_skips[j]
        if d1 == d2:
            result.append(d1); i += 1; j += 1
        elif d1 < d2:
            # Try to use skip pointer on p1
            if s1 is not None and p1_skips[s1][0] <= d2:
                i = s1
            else:
                i += 1
        else:
            if s2 is not None and p2_skips[s2][0] <= d1:
                j = s2
            else:
                j += 1
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    docs = {
        0: "information retrieval lucene index search",
        1: "boolean retrieval inverted index posting",
        2: "lucene search engine apache pylucene",
        3: "term document matrix boolean query",
        4: "inverted index compression postings list",
    }

    idx = InvertedIndex()
    idx.build(docs)

    print("=== Inverted Index ===")
    for term in ["information", "lucene", "index", "boolean"]:
        print(f"  '{term}': {idx.get_postings(term)}")

    print("\n=== Boolean Queries ===")
    print(f"  lucene AND index      : {idx.AND('lucene', 'index')}")
    print(f"  boolean OR lucene     : {idx.OR('boolean', 'lucene')}")
    print(f"  index AND NOT boolean : {idx.AND_NOT('index', 'boolean')}")

    print("\n=== Skip Pointers ===")
    postings = list(range(0, 20, 2))          # [0,2,4,...,18]
    skipped  = add_skip_pointers(postings)
    print(f"  Postings with skips: {skipped}")
