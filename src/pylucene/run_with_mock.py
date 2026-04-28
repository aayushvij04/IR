import sys
sys.path.append("/Users/aayushvij/Desktop/IR")
from pylucene_mock import inject_mock
inject_mock()

import indexing
import retrieval
import query_classes

print("Done testing!")
