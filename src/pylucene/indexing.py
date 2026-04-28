"""
pylucene/indexing.py
====================
Demonstrates PyLucene indexing:
  - Initializing the JVM
  - Creating an IndexWriter with custom Analyzer
  - Adding documents with various Field types:
      TextField   — full-text indexed, not stored as-is
      StringField — indexed as a single token (keyword), not analyzed
      StoredField — stored but not indexed (for retrieval/display)
      IntPoint    — numeric range-queryable integer field
      FloatPoint  — numeric range-queryable float field
  - Committing and closing the index
  - Reading back documents with DirectoryReader
"""

import lucene
from org.apache.lucene.store          import FSDirectory, ByteBuffersDirectory
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core  import WhitespaceAnalyzer, KeywordAnalyzer
from org.apache.lucene.index          import IndexWriter, IndexWriterConfig, DirectoryReader
from org.apache.lucene.document       import (Document, TextField, StringField,
                                               StoredField, IntPoint, FloatPoint,
                                               Field)
from org.apache.lucene.search         import IndexSearcher
from java.nio.file                    import Paths


# ---------------------------------------------------------------------------
# Sample movie dataset (simulating Wiki Movies)
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


# ---------------------------------------------------------------------------
# Helper: create a Lucene Document from a movie dict
# ---------------------------------------------------------------------------

def movie_to_document(movie: dict) -> Document:
    """
    Map a movie dict to a Lucene Document with appropriate field types.

    Field type selection guide:
      TextField   → full-text search (analyzed, tokenized)
      StringField → exact-match / keyword search (not tokenized)
      StoredField → retrieve value at query time (not indexed)
      IntPoint    → numeric range queries (e.g., year >= 2000)
      FloatPoint  → floating-point range queries (e.g., rating >= 8.5)
    """
    doc = Document()

    # Keyword (not analyzed) — for exact match
    doc.add(StringField("id",    str(movie["id"]),    Field.Store.YES))
    doc.add(StringField("genre", movie["genre"],       Field.Store.YES))

    # Full-text (analyzed with StandardAnalyzer)
    doc.add(TextField("title",   movie["title"],       Field.Store.YES))
    doc.add(TextField("plot",    movie["plot"],        Field.Store.YES))

    # Stored-only (for display, not searchable)
    doc.add(StoredField("year_display", str(movie["year"])))

    # Numeric fields (for range queries)
    doc.add(IntPoint("year",     movie["year"]))
    doc.add(FloatPoint("rating", movie["rating"]))

    # Store numeric values for retrieval
    doc.add(StoredField("year",   movie["year"]))
    doc.add(StoredField("rating", movie["rating"]))

    return doc


# ---------------------------------------------------------------------------
# Indexing function
# ---------------------------------------------------------------------------

def build_index(index_dir: str = None) -> object:
    """
    Build a Lucene index from the MOVIES dataset.

    Args:
        index_dir: Path to persist the index (FSDirectory).
                   If None, uses an in-memory ByteBuffersDirectory.

    Returns:
        The Lucene Directory object (open for reading).
    """
    # JVM must be initialized before any Lucene calls
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    # Choose storage: on-disk or in-memory
    if index_dir:
        directory = FSDirectory.open(Paths.get(index_dir))
    else:
        directory = ByteBuffersDirectory()

    analyzer = StandardAnalyzer()
    config   = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

    writer = IndexWriter(directory, config)

    for movie in MOVIES:
        doc = movie_to_document(movie)
        writer.addDocument(doc)
        print(f"  Indexed: [{movie['id']}] {movie['title']} ({movie['year']})")

    writer.commit()
    writer.close()
    print(f"\n  ✓ Index built with {len(MOVIES)} documents.")
    return directory


# ---------------------------------------------------------------------------
# Read back all documents
# ---------------------------------------------------------------------------

def read_all_documents(directory) -> None:
    """Iterate over all stored documents in the index."""
    reader   = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    num_docs = reader.numDocs()
    print(f"\n=== Index contains {num_docs} documents ===")
    for i in range(num_docs):
        doc = searcher.doc(i)
        print(f"  [{doc.get('id')}] {doc.get('title')} "
              f"| Genre: {doc.get('genre')} "
              f"| Year: {doc.get('year')} "
              f"| Rating: {doc.get('rating')}")
    reader.close()


# ---------------------------------------------------------------------------
# Custom Analyzer selection
# ---------------------------------------------------------------------------

def get_analyzer(name: str = "standard"):
    """
    Return a Lucene Analyzer by name.

    Options:
      'standard'   — lowercase, removes stop words, applies stemming
      'whitespace' — splits only on whitespace, preserves case
      'keyword'    — treats entire field as a single token
    """
    analyzers = {
        "standard":   StandardAnalyzer,
        "whitespace": WhitespaceAnalyzer,
        "keyword":    KeywordAnalyzer,
    }
    cls = analyzers.get(name.lower(), StandardAnalyzer)
    return cls()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PyLucene Indexing Demo ===\n")
    directory = build_index()      # in-memory index
    read_all_documents(directory)
