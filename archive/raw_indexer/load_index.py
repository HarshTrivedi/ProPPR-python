import pickle
import awesome_print as ap
import os
import re

# Config START #
index_path = "./index"
# Config END  #

direct_index = pickle.load(
    open(os.path.join(index_path, "direct_index.pickle"), "rb"))
inverted_index = pickle.load(
    open(os.path.join(index_path, "inverted_index.pickle"), "rb"))
document_meta_data = pickle.load(
    open(os.path.join(index_path, "document_meta_data.pickle"), "rb"))
collection_meta_data = pickle.load(
    open(os.path.join(index_path, "collection_meta_data.pickle"), "rb"))
