"""
index_compression.py
====================
Covers:
  - BSBI  (Block Sort-Based Indexing)
  - SPIMI (Single-Pass In-Memory Indexing)
  - Zipf's Law verification
  - Heaps' Law vocabulary size estimation
  - Variable-Byte Encoding (VBE) for postings compression
  - Gamma Encoding for gap-encoded postings
"""

import math
import heapq
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# 1. BSBI — Block Sort-Based Indexing (simplified simulation)
# ---------------------------------------------------------------------------

def bsbi_index(documents: dict, block_size: int = 3) -> dict:
    """
    Simulate BSBI: process documents in blocks, sort & write each block,
    then merge all block indices.

    Args:
        documents : {doc_id: text}
        block_size: number of docs per block

    Returns:
        merged inverted index {term: sorted list of doc_ids}
    """
    doc_items = list(documents.items())
    block_indices = []

    # --- Phase 1: build & sort each block ---
    for start in range(0, len(doc_items), block_size):
        block = doc_items[start: start + block_size]
        block_index = defaultdict(list)
        for doc_id, text in block:
            for token in text.lower().split():
                if doc_id not in block_index[token]:
                    block_index[token].append(doc_id)
        # Sort each postings list in the block
        for term in block_index:
            block_index[term].sort()
        block_indices.append(dict(block_index))

    # --- Phase 2: merge all block indices ---
    merged = defaultdict(list)
    for block in block_indices:
        for term, postings in block.items():
            merged[term].extend(postings)

    for term in merged:
        merged[term] = sorted(set(merged[term]))

    return dict(merged)


# ---------------------------------------------------------------------------
# 2. SPIMI — Single-Pass In-Memory Indexing
# ---------------------------------------------------------------------------

def spimi_index(token_stream: list[tuple]) -> dict:
    """
    SPIMI: single pass over a (term, doc_id) token stream.
    Directly appends to postings without sorting term-pairs first.

    Args:
        token_stream: list of (term, doc_id)

    Returns:
        inverted index {term: sorted list of doc_ids}
    """
    index = defaultdict(set)
    for term, doc_id in token_stream:
        index[term].add(doc_id)

    # Sort and deduplicate
    return {term: sorted(postings) for term, postings in sorted(index.items())}


# ---------------------------------------------------------------------------
# 3. Zipf's Law
# ---------------------------------------------------------------------------

def zipf_analysis(text: str) -> list[tuple]:
    """
    Verify Zipf's Law: frequency of the r-th most common term ~ C / r.

    Returns list of (rank, term, freq, expected_freq) sorted by rank.
    """
    tokens = text.lower().split()
    freq = Counter(tokens)
    sorted_terms = freq.most_common()

    most_common_freq = sorted_terms[0][1]
    result = []
    for rank, (term, f) in enumerate(sorted_terms, start=1):
        expected = most_common_freq / rank
        result.append((rank, term, f, round(expected, 2)))
    return result


# ---------------------------------------------------------------------------
# 4. Heaps' Law — vocabulary size estimation
# ---------------------------------------------------------------------------

def heaps_law(n_tokens: int, k: float = 44.0, b: float = 0.5) -> int:
    """
    Estimate vocabulary size M from collection size n_tokens.
    Heaps' Law: M = k * n^b

    Typical values: k ≈ 10–100, b ≈ 0.4–0.6
    """
    return int(k * (n_tokens ** b))


# ---------------------------------------------------------------------------
# 5. Variable-Byte Encoding (VBE)
# ---------------------------------------------------------------------------

def vbe_encode(numbers: list[int]) -> list[int]:
    """
    VBE-encode a list of positive integers (e.g., gap-encoded postings).
    Uses 7 bits per byte for data, MSB=1 signals the last byte of a number.
    """
    encoded = []
    for n in numbers:
        bytes_ = []
        while True:
            bytes_.insert(0, n % 128)
            n //= 128
            if n == 0:
                break
        bytes_[-1] += 128   # mark last byte
        encoded.extend(bytes_)
    return encoded


def vbe_decode(bytestream: list[int]) -> list[int]:
    """Decode a VBE-encoded byte stream back to integers."""
    numbers, n = [], 0
    for byte in bytestream:
        if byte < 128:
            n = 128 * n + byte
        else:
            n = 128 * n + (byte - 128)
            numbers.append(n)
            n = 0
    return numbers


def postings_to_gaps(postings: list[int]) -> list[int]:
    """Convert sorted postings to gap (delta) representation."""
    if not postings:
        return []
    gaps = [postings[0]]
    for i in range(1, len(postings)):
        gaps.append(postings[i] - postings[i - 1])
    return gaps


def gaps_to_postings(gaps: list[int]) -> list[int]:
    """Reconstruct postings from gap representation."""
    postings = []
    running = 0
    for g in gaps:
        running += g
        postings.append(running)
    return postings


# ---------------------------------------------------------------------------
# 6. Gamma Encoding (Elias gamma code)
# ---------------------------------------------------------------------------

def gamma_encode(n: int) -> str:
    """
    Elias gamma encoding of positive integer n.
    Offset = floor(log2(n)), unary prefix of (offset) zeros + '1' + offset bits.
    """
    if n < 1:
        raise ValueError("Gamma encoding requires n >= 1")
    offset = int(math.log2(n))
    unary  = "0" * offset + "1"
    remainder = bin(n)[2:][1:]          # offset-many bits after leading 1
    return unary + remainder


def gamma_decode(bits: str) -> tuple[int, str]:
    """Decode one gamma-encoded integer from the start of a bit string.
    Returns (value, remaining_bits)."""
    offset = 0
    while offset < len(bits) and bits[offset] == "0":
        offset += 1
    if offset >= len(bits):
        raise ValueError("Invalid gamma code")
    value_bits = "1" + bits[offset + 1: offset + 1 + offset]
    value = int(value_bits, 2) if value_bits else 1
    return value, bits[offset + 1 + offset:]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    docs = {
        1: "information retrieval index lucene search",
        2: "boolean retrieval inverted index posting",
        3: "lucene search engine pylucene apache",
        4: "term document matrix boolean query",
        5: "inverted index compression postings list",
        6: "ranking retrieval tf idf cosine similarity",
    }

    # BSBI
    bsbi_result = bsbi_index(docs, block_size=3)
    print("=== BSBI Index (first 5 terms) ===")
    for term in list(bsbi_result)[:5]:
        print(f"  {term}: {bsbi_result[term]}")

    # SPIMI
    stream = [(tok, did) for did, text in docs.items()
              for tok in text.lower().split()]
    spimi_result = spimi_index(stream)
    print("\n=== SPIMI Index (first 5 terms) ===")
    for term in list(spimi_result)[:5]:
        print(f"  {term}: {spimi_result[term]}")

    # Zipf
    corpus = " ".join(docs.values())
    zipf = zipf_analysis(corpus)
    print("\n=== Zipf's Law (top 8 terms) ===")
    print(f"  {'Rank':>4}  {'Term':<12}  {'Freq':>4}  {'Expected':>8}")
    for row in zipf[:8]:
        print(f"  {row[0]:>4}  {row[1]:<12}  {row[2]:>4}  {row[3]:>8}")

    # Heaps
    print(f"\n=== Heaps' Law ===")
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        print(f"  n={n:>9,}  estimated vocabulary: {heaps_law(n):>6,}")

    # VBE
    postings = [1, 5, 10, 20, 50, 100, 500]
    gaps     = postings_to_gaps(postings)
    encoded  = vbe_encode(gaps)
    decoded  = gaps_to_postings(vbe_decode(encoded))
    print(f"\n=== VBE Compression ===")
    print(f"  Original postings : {postings}")
    print(f"  Gaps              : {gaps}")
    print(f"  VBE bytes         : {encoded}  ({len(encoded)} bytes)")
    print(f"  Decoded postings  : {decoded}")

    # Gamma
    print(f"\n=== Gamma Encoding ===")
    for n in [1, 3, 9, 13, 24]:
        code = gamma_encode(n)
        print(f"  gamma({n:>2}) = '{code}'")
