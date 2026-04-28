# Information Retrieval with PyLucene

A comprehensive repository on **Information Retrieval (IR)** — covering core IR algorithms from scratch and practical **PyLucene** integration for real-world search systems.

📄 **[Full Report (PDF)](report/IR_Report_Aayush_Vandan.pdf)** — A detailed 4–5 page report on IR fundamentals and PyLucene.

---

## 📂 Repository Structure

```
IR/
├── src/
│   ├── boolean_retrieval.py      # Inverted Index, Boolean ops, Skip Pointers
│   ├── index_compression.py      # BSBI, SPIMI, Zipf's Law, Heaps' Law, Compression
│   ├── ranking_models.py         # TF-IDF, VSM, BM25, BM25F, Cosine Similarity
│   ├── language_models.py        # Unigram LM, Jelinek-Mercer, Dirichlet Smoothing
│   ├── evaluation.py             # Precision, Recall, AP, MAP, nDCG, MRR, Kappa
│   ├── web_search.py             # PageRank, HITS, Shingling
│   ├── advanced_ir.py            # LSI, Word Embeddings, BERT-based Retrieval
│   └── pylucene/
│       ├── indexing.py           # PyLucene indexing, Document & Field options
│       ├── retrieval.py          # PyLucene search & scoring
│       └── query_classes.py      # TermQuery, PhraseQuery, BooleanQuery, FuzzyQuery…
└── report/
    ├── IR_Report.tex                 # LaTeX source
    └── IR_Report_Aayush_Vandan.pdf   # Compiled report
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy scipy scikit-learn rank_bm25
# PyLucene — requires JCC + Java: https://lucene.apache.org/pylucene/install.html
```

### Running Examples
```bash
# Core IR algorithms
python src/boolean_retrieval.py
python src/ranking_models.py
python src/evaluation.py

# PyLucene (requires PyLucene installed)
python src/pylucene/indexing.py
python src/pylucene/retrieval.py
python src/pylucene/query_classes.py
```

---

## 🔍 Topics Covered

| Module | Topics |
|--------|--------|
| `boolean_retrieval.py` | Term-Document Matrix, Inverted Index, AND/OR/NOT, Skip Pointers |
| `index_compression.py` | BSBI, SPIMI, Zipf's & Heaps' Law, VBE/Gamma Encoding |
| `ranking_models.py` | TF-IDF, Cosine Similarity, VSM, BM25, BM25F, BM25+ |
| `language_models.py` | Unigram LM, JM Smoothing, Dirichlet Smoothing, KLD |
| `evaluation.py` | P@K, R-Prec, AP, MAP, MRR, nDCG, Kappa |
| `web_search.py` | PageRank (random surfer), HITS, Shingling/Jaccard |
| `advanced_ir.py` | LSI/SVD, Word2Vec-style embeddings, BERT reranking |
| `pylucene/indexing.py` | IndexWriter, Document, TextField, StringField, Analyzers |
| `pylucene/retrieval.py` | IndexSearcher, BM25Similarity, TopDocs |
| `pylucene/query_classes.py` | All PyLucene query types with examples |

---

> **Repository**: https://github.com/aayushvij04/IR
