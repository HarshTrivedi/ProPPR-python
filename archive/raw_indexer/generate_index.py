import awesome_print as ap
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import pickle
import xmltodict
import csv
import sys
import nltk
import os
import re

# Config START #
document_tag_to_index = "title-abstract-content"
documents_path = "./selected_documents"
index_path = "./index"
# Config END  #

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = PorterStemmer()
stopwords_dictionary = {}
english_stopwords = [
    "a", "about", "above", "above", "across", "after", "afterwards", "again",
    "against", "all", "almost", "alone", "along", "already", "also", "although",
    "always", "am", "among", "amongst", "amoungst", "amount", "an", "and",
    "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere",
    "are", "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom",
    "but", "by", "call", "can", "cannot", "cant", "co", "con", "could",
    "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due",
    "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere",
    "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything",
    "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire",
    "first", "five", "for", "former", "formerly", "forty", "found", "four",
    "from", "front", "full", "further", "get", "give", "go", "had", "has",
    "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby",
    "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how",
    "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into",
    "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least",
    "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill",
    "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my",
    "myself", "name", "namely", "neither", "never", "nevertheless", "next",
    "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now",
    "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or",
    "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over",
    "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same",
    "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she",
    "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
    "somehow", "someone", "something", "sometime", "sometimes", "somewhere",
    "still", "such", "system", "take", "ten", "than", "that", "the", "their",
    "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thickv", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within",
    "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves",
    "the"
]
english_stopwords += stopwords.words('english')
for stopword in english_stopwords:
    stopwords_dictionary[stopword] = True

inverted_index = {}
# term -> {docid1 -> freq1 , docid2 -> freq2 ...}
direct_index = {}
# docid -> {term1 -> freq1 , term2 -> freq2 ...}

document_meta_data = {}
# docid -> {words_used -> Num 1, vocabulary_size -> Num 2}

collection_meta_data = {
    "words_used": 0,
    "vocabulary_size": 0,
    "number_of_documents": 0,
    "average_words_used_per_document": 0
}
# {words_used -> Num 1, vocabulary_size -> Num 2, number_of_documents -> Num 2, average_document_size -> Num 4}

count = 0
for dirpath, dirs, files in os.walk(documents_path):
    for filename in files:
        document_id = int(filename.replace(".xml", ""))
        full_path = os.path.join(dirpath, filename)
        with open(full_path) as fd:
            documents_dictionary = xmltodict.parse(fd.read())
            for tag in document_tag_to_index.split("-"):

                text = documents_dictionary['document'][tag]
                if text != None:
                    sentences = sentence_tokenizer.tokenize(text)
                    for sentence in sentences:
                        word_tokens = word_tokenize(sentence)
                        word_tokens = [
                            stemmer.stem(word_token.lower())
                            for word_token in word_tokens
                            if not stopwords_dictionary.
                            get(word_token.lower(), False) and re.match(
                                '^[ _\w-]+$', word_token.lower()) is not None
                        ]
                        # ap(word_tokens)
                        for word_token in word_tokens:
                            word_token = word_token.encode("utf-8")
                            if inverted_index.get(word_token, None) is None:
                                inverted_index[word_token] = {}
                            inverted_index[word_token][
                                document_id] = inverted_index[word_token].get(
                                    document_id, 0) + 1

                            if direct_index.get(document_id, None) is None:
                                direct_index[document_id] = {}
                            direct_index[document_id][
                                word_token] = direct_index[document_id].get(
                                    word_token, 0) + 1

                            collection_meta_data["words_used"] += 1

            document_meta_data[document_id] = {}
            document_meta_data[document_id]["words_used"] = len(
                direct_index[document_id])
            document_meta_data[document_id]["vocabulary_size"] = len(
                set(direct_index[document_id]))
        count += 1
        print count
        collection_meta_data["number_of_documents"] += 1

collection_meta_data["average_words_used_per_document"] = (
    float(collection_meta_data["words_used"]) /
    (float(collection_meta_data["number_of_documents"])))

with open(os.path.join(index_path, "direct_index"), 'w') as file_object:
    for document_id in direct_index:
        term_frequencies = direct_index[document_id].items()
        post = '|'.join([
            '='.join(map(str, term_frequency))
            for term_frequency in term_frequencies
        ])
        pre = str(document_id)
        line = '|'.join([pre, post])
        file_object.write(line + "\n")
pickle.dump(direct_index,
            open(os.path.join(index_path, "direct_index.pickle"), "wb"))

with open(os.path.join(index_path, "inverted_index"), 'w') as file_object:
    for term in inverted_index:
        document_frequencies = inverted_index[term].items()
        post = '|'.join([
            '='.join(map(str, document_frequency))
            for document_frequency in document_frequencies
        ])
        pre = str(term)
        line = '|'.join([pre, post])
        file_object.write(line + "\n")
pickle.dump(inverted_index,
            open(os.path.join(index_path, "inverted_index.pickle"), "wb"))

with open(os.path.join(index_path, "document_meta_data"), 'w') as file_object:
    for document_id in document_meta_data:
        words_used = document_meta_data[document_id]["words_used"]
        vocabulary_size = document_meta_data[document_id]["vocabulary_size"]
        post = "words_used={}|vocabulary_size={}".format(
            words_used, vocabulary_size)
        pre = str(document_id)
        line = '|'.join([pre, post])
        file_object.write(line + "\n")
pickle.dump(document_meta_data,
            open(os.path.join(index_path, "document_meta_data.pickle"), "wb"))

with open(os.path.join(index_path, "collection_meta_data"), 'w') as file_object:
    words_used = collection_meta_data["words_used"]
    vocabulary_size = collection_meta_data["vocabulary_size"]
    number_of_documents = collection_meta_data["number_of_documents"]
    average_words_used_per_document = collection_meta_data[
        "average_words_used_per_document"]
    line = "words_used={}|vocabulary_size={}|number_of_documents={}|average_words_used_per_document={}".format(
        words_used, vocabulary_size, number_of_documents,
        average_words_used_per_document)
    file_object.write(line + "\n")
pickle.dump(collection_meta_data,
            open(os.path.join(index_path, "collection_meta_data.pickle"), "wb"))
