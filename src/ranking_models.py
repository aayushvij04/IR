"""
ranking_models.py
=================
Implements ranked retrieval models:
  - TF, scaled TF, IDF, TF-IDF weighting
  - Vector Space Model (VSM) with cosine similarity
  - SMART notation
  - BM25, BM25F, BM25+
"""

import math
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# 1. TF-IDF Weighting
# ---------------------------------------------------------------------------

def tf_raw(term: str, doc_tokens: list[str]) -> int:
    return doc_tokens.count(term)

def tf_log(term: str, doc_tokens: list[str]) -> float:
    raw = tf_raw(term, doc_tokens)
    return 1 + math.log10(raw) if raw > 0 else 0.0

def tf_augmented(term: str, doc_tokens: list[str], K: float = 0.5) -> float:
    """Augmented (normalized) TF to prevent long-doc bias."""
    raw = tf_raw(term, doc_tokens)
    max_tf = max(doc_tokens.count(t) for t in set(doc_tokens)) if doc_tokens else 1
    return K + (1 - K) * raw / max_tf if max_tf > 0 else 0.0

def idf(term: str, all_docs: list[list[str]]) -> float:
    N = len(all_docs)
    df = sum(1 for doc in all_docs if term in doc)
    return math.log10(N / df) if df > 0 else 0.0

def tfidf_vector(query_terms: list[str], doc_tokens: list[str],
                 all_docs: list[list[str]]) -> dict[str, float]:
    """Compute TF-IDF weights for query terms w.r.t. a document."""
    return {
        t: tf_log(t, doc_tokens) * idf(t, all_docs)
        for t in query_terms
    }


# ---------------------------------------------------------------------------
# 2. Vector Space Model (VSM)
# ---------------------------------------------------------------------------

class VSM:
    """
    Vector Space Model with TF-IDF weighting and cosine similarity.
    SMART notation: lnc (docs) · ltc (query) by default.
    """

    def __init__(self, documents: dict[int, str]):
        self.doc_ids = list(documents.keys())
        self.doc_tokens: dict[int, list[str]] = {
            did: text.lower().split() for did, text in documents.items()
        }
        self.all_tokens = list(self.doc_tokens.values())
        self._build_index()

    def _build_index(self):
        """Pre-compute TF-IDF document vectors."""
        all_terms = set(t for tokens in self.all_tokens for t in tokens)
        self.idf_cache: dict[str, float] = {
            t: idf(t, self.all_tokens) for t in all_terms
        }
        self.doc_vectors: dict[int, dict[str, float]] = {}
        for did, tokens in self.doc_tokens.items():
            vec = {}
            for term in set(tokens):
                vec[term] = tf_log(term, tokens) * self.idf_cache.get(term, 0)
            # L2 normalize (lnc → cosine normalization)
            norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
            self.doc_vectors[did] = {t: v / norm for t, v in vec.items()}

    def _query_vector(self, query: str) -> dict[str, float]:
        """Build ltc query vector."""
        tokens = query.lower().split()
        raw_vec = {}
        for term in set(tokens):
            raw_vec[term] = tf_log(term, tokens) * self.idf_cache.get(term, 0)
        norm = math.sqrt(sum(v ** 2 for v in raw_vec.values())) or 1.0
        return {t: v / norm for t, v in raw_vec.items()}

    def cosine_similarity(self, vec1: dict, vec2: dict) -> float:
        """Dot product of two L2-normalised vectors = cosine similarity."""
        return sum(vec1.get(t, 0) * vec2.get(t, 0)
                   for t in vec1 if t in vec2)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Return top-k documents ranked by cosine similarity."""
        q_vec = self._query_vector(query)
        scores = [
            (did, self.cosine_similarity(q_vec, self.doc_vectors[did]))
            for did in self.doc_ids
        ]
        return sorted(scores, key=lambda x: -x[1])[:top_k]


# ---------------------------------------------------------------------------
# 3. BM25
# ---------------------------------------------------------------------------

class BM25:
    """
    Okapi BM25 ranking function.
    Score(D, Q) = Σ IDF(qi) * (tf * (k1+1)) / (tf + k1*(1 - b + b*|D|/avgdl))
    """

    def __init__(self, documents: dict[int, str],
                 k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.doc_ids = list(documents.keys())
        self.doc_tokens = {
            did: text.lower().split() for did, text in documents.items()
        }
        self.N = len(documents)
        self.avgdl = sum(len(t) for t in self.doc_tokens.values()) / self.N
        self._build_df()

    def _build_df(self):
        self.df: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens.values():
            for term in set(tokens):
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _score_doc(self, query_terms: list[str], did: int) -> float:
        tokens = self.doc_tokens[did]
        dl = len(tokens)
        tf_map = Counter(tokens)
        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            idf_val = self._idf(term)
            numerator   = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf_val * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        terms = query.lower().split()
        scores = [(did, self._score_doc(terms, did)) for did in self.doc_ids]
        return sorted(scores, key=lambda x: -x[1])[:top_k]


# ---------------------------------------------------------------------------
# 4. BM25+ (Lower-bound on TF saturation)
# ---------------------------------------------------------------------------

class BM25Plus(BM25):
    """
    BM25+ adds a lower bound delta (δ) to the TF term to prevent
    zero contribution for matching terms in long documents.
    """

    def __init__(self, documents: dict[int, str],
                 k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        super().__init__(documents, k1, b)
        self.delta = delta

    def _score_doc(self, query_terms: list[str], did: int) -> float:
        tokens = self.doc_tokens[did]
        dl = len(tokens)
        tf_map = Counter(tokens)
        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            idf_val = self._idf(term)
            tf_norm = tf / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            score += idf_val * (self.delta + tf_norm * (self.k1 + 1))
        return score


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    documents = {
        1: "information retrieval is the science of searching large collections",
        2: "lucene provides full text search with tf idf and bm25 ranking",
        3: "boolean retrieval uses an inverted index for exact term matching",
        4: "bm25 is a probabilistic ranking model used in modern search engines",
        5: "cosine similarity measures the angle between two tf idf vectors",
        6: "language models use smoothing to handle zero frequency terms in retrieval",
    }

    query = "bm25 retrieval ranking"

    # VSM
    vsm = VSM(documents)
    print("=== VSM Results ===")
    for did, score in vsm.search(query):
        print(f"  doc {did}: {score:.4f} | {documents[did][:55]}")

    # BM25
    bm25 = BM25(documents)
    print("\n=== BM25 Results ===")
    for did, score in bm25.search(query):
        print(f"  doc {did}: {score:.4f} | {documents[did][:55]}")

    # BM25+
    bm25p = BM25Plus(documents)
    print("\n=== BM25+ Results ===")
    for did, score in bm25p.search(query):
        print(f"  doc {did}: {score:.4f} | {documents[did][:55]}")

    # Manual TF-IDF example
    all_tokens = [t.lower().split() for t in documents.values()]
    doc1_tokens = documents[1].lower().split()
    qterms = ["information", "retrieval"]
    vec = tfidf_vector(qterms, doc1_tokens, all_tokens)
    print(f"\n=== TF-IDF vector for doc 1, query {qterms} ===")
    for t, w in vec.items():
        print(f"  {t}: {w:.4f}")
