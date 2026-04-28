"""
pylucene/retrieval.py
=====================
Demonstrates PyLucene search & retrieval:
  - IndexSearcher setup
  - BM25Similarity vs ClassicSimilarity (TF-IDF)
  - Executing queries and reading TopDocs
  - Displaying scored hits with stored fields
  - Using QueryParser for natural-language queries
  - Rocchio-based query expansion in PyLucene
"""

import lucene
from org.apache.lucene.store            import ByteBuffersDirectory
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index            import DirectoryReader, IndexWriter, IndexWriterConfig
from org.apache.lucene.search           import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.search.similarities import BM25Similarity, ClassicSimilarity
from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser
from org.apache.lucene.document         import (Document, TextField, StringField,
                                                 StoredField, IntPoint, FloatPoint, Field)
from org.apache.lucene.search           import TermQuery, BoostQuery
from org.apache.lucene.index            import Term


# ---------------------------------------------------------------------------
# Re-use the movie dataset from indexing.py
# ---------------------------------------------------------------------------

MOVIES = [
    {"id": 1, "title": "The Dark Knight",
     "plot": "Batman fights the Joker in Gotham City to restore order and justice.",
     "genre": "Action", "year": 2008, "rating": 9.0},
    {"id": 2, "title": "Inception",
     "plot": "A thief who steals corporate secrets through dream-sharing technology.",
     "genre": "Sci-Fi", "year": 2010, "rating": 8.8},
    {"id": 3, "title": "The Shawshank Redemption",
     "plot": "Two imprisoned men bond over years finding solace and eventual redemption.",
     "genre": "Drama", "year": 1994, "rating": 9.3},
    {"id": 4, "title": "Interstellar",
     "plot": "Astronauts travel through a wormhole in search of a new home for humanity.",
     "genre": "Sci-Fi", "year": 2014, "rating": 8.6},
    {"id": 5, "title": "Pulp Fiction",
     "plot": "Several interconnected stories of crime in Los Angeles.",
     "genre": "Crime", "year": 1994, "rating": 8.9},
]


def _build_test_index() -> tuple:
    """Build an in-memory index for demo purposes. Returns (directory, analyzer)."""
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    directory = ByteBuffersDirectory()
    analyzer  = StandardAnalyzer()
    config    = IndexWriterConfig(analyzer)
    writer    = IndexWriter(directory, config)
    for m in MOVIES:
        doc = Document()
        doc.add(StringField("id",    str(m["id"]),  Field.Store.YES))
        doc.add(TextField("title",   m["title"],    Field.Store.YES))
        doc.add(TextField("plot",    m["plot"],     Field.Store.YES))
        doc.add(StringField("genre", m["genre"],    Field.Store.YES))
        doc.add(StoredField("year",  str(m["year"])))
        doc.add(StoredField("rating",str(m["rating"])))
        doc.add(IntPoint("year_idx", m["year"]))
        doc.add(FloatPoint("rating_idx", m["rating"]))
        writer.addDocument(doc)
    writer.commit(); writer.close()
    return directory, analyzer


# ---------------------------------------------------------------------------
# 1. Basic search with BM25 (default Lucene similarity)
# ---------------------------------------------------------------------------

def search_bm25(directory, analyzer, query_str: str, top_k: int = 5):
    """
    Search using BM25Similarity (Lucene's default since v8).
    BM25 parameters: k1=1.2, b=0.75 (can be tuned).
    """
    reader   = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity(1.2, 0.75))   # k1, b

    parser = QueryParser("plot", analyzer)
    query  = parser.parse(query_str)

    print(f"\n=== BM25 Search: '{query_str}' ===")
    print(f"  Parsed query: {query}")
    top_docs = searcher.search(query, top_k)
    print(f"  Total hits: {top_docs.totalHits.value}")

    for hit in top_docs.scoreDocs:
        doc = searcher.doc(hit.doc)
        print(f"  Score={hit.score:.4f} | [{doc.get('id')}] "
              f"{doc.get('title')} ({doc.get('year')})")
    reader.close()
    return top_docs


# ---------------------------------------------------------------------------
# 2. Search with Classic TF-IDF similarity
# ---------------------------------------------------------------------------

def search_tfidf(directory, analyzer, query_str: str, top_k: int = 5):
    """
    Search using ClassicSimilarity (Lucene's legacy TF-IDF).
    Useful for comparing BM25 vs TF-IDF on the same corpus.
    """
    reader   = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(ClassicSimilarity())

    parser = QueryParser("plot", analyzer)
    query  = parser.parse(query_str)

    print(f"\n=== TF-IDF Search: '{query_str}' ===")
    top_docs = searcher.search(query, top_k)
    for hit in top_docs.scoreDocs:
        doc = searcher.doc(hit.doc)
        print(f"  Score={hit.score:.4f} | [{doc.get('id')}] {doc.get('title')}")
    reader.close()


# ---------------------------------------------------------------------------
# 3. Multi-field search (title + plot)
# ---------------------------------------------------------------------------

def search_multifield(directory, analyzer, query_str: str, top_k: int = 5):
    """
    Search across multiple fields (title + plot) with field boosts.
    Title matches are boosted 2× over plot matches.
    """
    reader   = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)

    # Field boost: title gets 2×, plot gets 1×
    boosts = {"title": 2.0, "plot": 1.0}
    parser = MultiFieldQueryParser(
        ["title", "plot"], analyzer,
        {k: float(v) for k, v in boosts.items()}
    )
    query = parser.parse(query_str)

    print(f"\n=== Multi-field Search (title^2 + plot): '{query_str}' ===")
    top_docs = searcher.search(query, top_k)
    for hit in top_docs.scoreDocs:
        doc = searcher.doc(hit.doc)
        print(f"  Score={hit.score:.4f} | [{doc.get('id')}] {doc.get('title')}")
    reader.close()


# ---------------------------------------------------------------------------
# 4. Explain scoring (debugging relevance)
# ---------------------------------------------------------------------------

def explain_score(directory, analyzer, query_str: str, doc_idx: int = 0):
    """
    Use IndexSearcher.explain() to understand how Lucene scored a document.
    Invaluable for debugging retrieval quality.
    """
    reader   = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity())

    parser   = QueryParser("plot", analyzer)
    query    = parser.parse(query_str)
    top_docs = searcher.search(query, 10)

    if top_docs.scoreDocs:
        hit         = top_docs.scoreDocs[doc_idx]
        explanation = searcher.explain(query, hit.doc)
        doc         = searcher.doc(hit.doc)
        print(f"\n=== Score Explanation for '{doc.get('title')}' ===")
        print(explanation.toString())
    reader.close()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    directory, analyzer = _build_test_index()

    search_bm25(directory, analyzer, "space travel wormhole humanity")
    search_tfidf(directory, analyzer, "crime city justice")
    search_multifield(directory, analyzer, "redemption prison bond")
    explain_score(directory, analyzer, "dream technology corporate secrets")
