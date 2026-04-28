"""
pylucene/query_classes.py
=========================
Demonstrates every major PyLucene Query class:

  TermQuery          — exact single-term match
  PhraseQuery        — ordered phrase / slop-based proximity
  TermRangeQuery     — alphabetical / lexicographic range
  IntPoint.newRange  — numeric (integer) range query
  FloatPoint.newRange— numeric (float) range query
  PrefixQuery        — terms starting with a prefix
  BooleanQuery       — AND / OR / NOT combinations of sub-queries
  WildcardQuery      — ? (single char) and * (zero+ chars) wildcards
  FuzzyQuery         — edit-distance (Levenshtein) approximate matching
  MatchAllDocsQuery  — retrieve every document in the index
  BoostQuery         — weight a sub-query by a boost factor

Each function prints the results with Lucene scores so you can compare
how the ranking changes across query types.
"""

import lucene
from org.apache.lucene.store            import ByteBuffersDirectory
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index            import (DirectoryReader, IndexWriter,
                                                IndexWriterConfig, Term)
from org.apache.lucene.document         import (Document, TextField, StringField,
                                                StoredField, IntPoint, FloatPoint,
                                                Field)
from org.apache.lucene.search           import (IndexSearcher, TermQuery,
                                                PhraseQuery, BooleanQuery,
                                                BooleanClause, WildcardQuery,
                                                FuzzyQuery, MatchAllDocsQuery,
                                                BoostQuery, TermRangeQuery)
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.util             import BytesRef
from org.apache.lucene.queryparser.classic import QueryParser


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

MOVIES = [
    {"id": "1", "title": "The Dark Knight",
     "plot": "Batman fights the Joker in Gotham City to restore order and justice",
     "genre": "action", "year": 2008, "rating": 9.0},
    {"id": "2", "title": "Inception",
     "plot": "A thief steals corporate secrets through a dream sharing technology",
     "genre": "scifi", "year": 2010, "rating": 8.8},
    {"id": "3", "title": "The Shawshank Redemption",
     "plot": "Two imprisoned men bond over years finding solace and eventual redemption",
     "genre": "drama", "year": 1994, "rating": 9.3},
    {"id": "4", "title": "Interstellar",
     "plot": "Astronauts travel through a wormhole in search of a new home for humanity",
     "genre": "scifi", "year": 2014, "rating": 8.6},
    {"id": "5", "title": "Pulp Fiction",
     "plot": "Interconnected stories of crime and redemption in Los Angeles",
     "genre": "crime", "year": 1994, "rating": 8.9},
    {"id": "6", "title": "The Godfather",
     "plot": "The aging patriarch of an organized crime dynasty transfers control to his son",
     "genre": "crime", "year": 1972, "rating": 9.2},
    {"id": "7", "title": "Schindler's List",
     "plot": "A businessman saves more than a thousand Jewish lives during the Holocaust",
     "genre": "drama", "year": 1993, "rating": 9.0},
]


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index():
    """Initialize JVM, build and return an in-memory Lucene index."""
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    directory = ByteBuffersDirectory()
    analyzer  = StandardAnalyzer()
    config    = IndexWriterConfig(analyzer)
    writer    = IndexWriter(directory, config)

    for m in MOVIES:
        doc = Document()
        doc.add(StringField("id",     m["id"],          Field.Store.YES))
        doc.add(TextField  ("title",  m["title"],        Field.Store.YES))
        doc.add(TextField  ("plot",   m["plot"],         Field.Store.YES))
        doc.add(StringField("genre",  m["genre"],        Field.Store.YES))
        doc.add(IntPoint   ("year",   m["year"]))
        doc.add(FloatPoint ("rating", m["rating"]))
        doc.add(StoredField("year_s", str(m["year"])))
        doc.add(StoredField("rating_s", str(m["rating"])))
        writer.addDocument(doc)

    writer.commit()
    writer.close()
    return directory, analyzer


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _run_query(searcher, query, label: str, top_k: int = 5):
    """Execute a query, print results table."""
    top_docs = searcher.search(query, top_k)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Query  : {query}")
    print(f"  Hits   : {top_docs.totalHits.value}")
    print(f"  {'Score':>7}  {'ID':>3}  {'Title'}")
    for hit in top_docs.scoreDocs:
        doc = searcher.doc(hit.doc)
        print(f"  {hit.score:>7.4f}  {doc.get('id'):>3}  {doc.get('title')}")


# ---------------------------------------------------------------------------
# Query demonstrations
# ---------------------------------------------------------------------------

def demo_term_query(searcher):
    """
    TermQuery — match documents containing an exact term (after analysis).
    Use on analyzed fields; the term must match post-analysis form.
    """
    query = TermQuery(Term("plot", "crime"))
    _run_query(searcher, query, "TermQuery  →  plot:crime")


def demo_phrase_query(searcher):
    """
    PhraseQuery — match an ordered phrase.
    slop=0  → exact phrase
    slop=N  → terms may be up to N positions apart (proximity search)
    """
    # Exact phrase
    exact = (PhraseQuery.Builder()
             .add(Term("plot", "corporate"))
             .add(Term("plot", "secrets"))
             .build())
    _run_query(searcher, exact, "PhraseQuery (slop=0)  →  'corporate secrets'")

    # Proximity (slop = 3)
    proximity = (PhraseQuery.Builder()
                 .add(Term("plot", "crime"))
                 .add(Term("plot", "redemption"))
                 .setSlop(3)
                 .build())
    _run_query(searcher, proximity, "PhraseQuery (slop=3)  →  'crime ... redemption'")


def demo_term_range_query(searcher):
    """
    TermRangeQuery — lexicographic range on a String/Text field.
    Useful for date strings, genre alphabetical ranges, etc.
    """
    # Genres alphabetically between 'crime' and 'drama' inclusive
    query = TermRangeQuery.newStringRange("genre", "crime", "drama",
                                         True, True)  # inclusive both ends
    _run_query(searcher, query,
               "TermRangeQuery  →  genre:[crime TO drama]")


def demo_numeric_range_query(searcher):
    """
    IntPoint.newRangeQuery / FloatPoint.newRangeQuery
    For indexed numeric fields — much faster than TermRangeQuery on strings.
    """
    # Movies released between 1990 and 2000
    year_q = IntPoint.newRangeQuery("year", 1990, 2000)
    _run_query(searcher, year_q,
               "IntPoint.newRangeQuery  →  year:[1990 TO 2000]")

    # Movies with rating >= 9.0
    rating_q = FloatPoint.newRangeQuery("rating", 9.0, Float.MAX_VALUE)
    _run_query(searcher, rating_q,
               "FloatPoint.newRangeQuery  →  rating:[9.0 TO *]")


def demo_prefix_query(searcher):
    """
    PrefixQuery — match all terms that begin with a given prefix.
    Efficient for autocomplete / type-ahead search.
    """
    from org.apache.lucene.search import PrefixQuery
    query = PrefixQuery(Term("plot", "inter"))
    _run_query(searcher, query, "PrefixQuery  →  plot:inter*")


def demo_wildcard_query(searcher):
    """
    WildcardQuery — '?' matches any single character, '*' matches zero or more.
    Warning: leading wildcards (*term) are expensive — disable in production.
    """
    query = WildcardQuery(Term("plot", "r?dempt*"))
    _run_query(searcher, query, "WildcardQuery  →  plot:r?dempt*")


def demo_fuzzy_query(searcher):
    """
    FuzzyQuery — approximate matching using Levenshtein (edit) distance.
    maxEdits=1 (default) or 2 — handles typos, misspellings.
    """
    query = FuzzyQuery(Term("plot", "astronawts"), 2)   # typo with 2 edits
    _run_query(searcher, query,
               "FuzzyQuery (maxEdits=2)  →  plot:astronawts~2")


def demo_boolean_query(searcher, analyzer):
    """
    BooleanQuery — combine multiple queries with:
      MUST    (AND)  — document must match this clause
      SHOULD  (OR)   — document should match (boosts score)
      MUST_NOT(NOT)  — document must NOT match this clause
    """
    # (plot contains 'crime' OR 'redemption') AND (genre is NOT 'action')
    builder = BooleanQuery.Builder()

    crime_q     = TermQuery(Term("plot", "crime"))
    redempt_q   = TermQuery(Term("plot", "redemption"))
    action_q    = TermQuery(Term("genre", "action"))

    builder.add(crime_q,   BooleanClause.Occur.SHOULD)
    builder.add(redempt_q, BooleanClause.Occur.SHOULD)
    builder.add(action_q,  BooleanClause.Occur.MUST_NOT)

    query = builder.build()
    _run_query(searcher, query,
               "BooleanQuery  →  (crime OR redemption) NOT genre:action")


def demo_match_all(searcher):
    """
    MatchAllDocsQuery — returns every document in the index.
    Typically used as a base for filtering (e.g., combined with BooleanQuery).
    """
    query = MatchAllDocsQuery()
    _run_query(searcher, query, "MatchAllDocsQuery  →  *:*", top_k=7)


def demo_boost_query(searcher):
    """
    BoostQuery — multiply a sub-query's score by a boost factor.
    Useful for promoting certain fields or sources.
    """
    title_q   = TermQuery(Term("title", "redemption"))
    plot_q    = TermQuery(Term("plot",  "redemption"))

    boosted_title = BoostQuery(title_q, 3.0)   # title match counts 3×
    builder = BooleanQuery.Builder()
    builder.add(boosted_title, BooleanClause.Occur.SHOULD)
    builder.add(plot_q,        BooleanClause.Occur.SHOULD)
    query = builder.build()
    _run_query(searcher, query,
               "BoostQuery  →  title:redemption^3.0 OR plot:redemption")


def demo_query_parser(directory, analyzer):
    """
    QueryParser — parse a human-readable query string into a Lucene Query.
    Supports: AND, OR, NOT, field:, phrase "", range [x TO y], wildcard *.
    """
    reader   = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)

    parser = QueryParser("plot", analyzer)

    examples = [
        'crime AND redemption',
        '"corporate secrets"',
        'title:interstellar OR title:inception',
        'plot:astron*',
        'plot:dreaming~1',       # fuzzy via query parser
    ]
    print(f"\n{'='*60}")
    print("  QueryParser examples")
    for q_str in examples:
        q = parser.parse(q_str)
        hits = searcher.search(q, 3).totalHits.value
        print(f"  '{q_str}'  →  {q}  [{hits} hit(s)]")
    reader.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from java.lang import Float

    print("Building index …")
    directory, analyzer = build_index()

    reader   = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity())

    demo_term_query(searcher)
    demo_phrase_query(searcher)
    demo_term_range_query(searcher)
    demo_numeric_range_query(searcher)
    demo_prefix_query(searcher)
    demo_wildcard_query(searcher)
    demo_fuzzy_query(searcher)
    demo_boolean_query(searcher, analyzer)
    demo_match_all(searcher)
    demo_boost_query(searcher)

    reader.close()

    demo_query_parser(directory, analyzer)
