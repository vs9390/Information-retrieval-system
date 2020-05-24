from bs4 import BeautifulSoup
from datetime import datetime
from collections import Counter
import nltk
import numpy as np
from scipy import spatial
import json
from pprint import pprint as pp

class InvertedIndex:

    # Constructor method
    def __init__(self):
        self.corpus_file_path = "wiki_00"
        self.documents_dict = {}
        self.bag_of_words = {}
        self.tf_idf_vector = {}
        self.doc_titles = {}
        self.cache_vector = None
        self.init_index()

    # Method which checks if there is any cache already built so that it can load index from cache, if not then proceed to build index from corpus
    def init_index(self):
        try:
            with open('cache/cache_vector.json', 'r') as cache_vector_file:
                self.cache_vector = json.loads(cache_vector_file.read())
            with open('cache/documents.json', 'r') as docuemnts_file:
                self.documents_dict = json.loads(docuemnts_file.read())
            with open('cache/bag_of_words.json', 'r') as bag_of_words_file:
                self.bag_of_words = json.loads(bag_of_words_file.read())
            with open('cache/doc_titles.json', 'r') as doc_titles_file:
                self.doc_titles = json.loads(doc_titles_file.read())
        except:
            pass

        if self.cache_vector is None:
            self.process_corpus_file()
            self.prepare_bag_of_words()
            self.build_index()

    # Read corpus file and build docuemnt_dict and doc_titles for later usases
    def process_corpus_file(self):
        print(datetime.now().strftime("%H:%M:%S") + ": Processing Corpus...")
        with open(self.corpus_file_path, 'r') as corpus_file:
            corpus_soup = BeautifulSoup(corpus_file.read(), 'html.parser')
            for doc in corpus_soup.find_all('doc'):
                self.documents_dict[int(doc['id'])] = doc.get_text()
                self.doc_titles[int(doc['id'])] = doc['title']
        print(datetime.now().strftime("%H:%M:%S") + ": Processed Corpus...")

    # Prerpare the bag of words with document frequency of each token
    def prepare_bag_of_words(self):
        print(datetime.now().strftime("%H:%M:%S") + ": Preparing bag of words...")
        for doc_id, doc_text in self.documents_dict.items():
            tokens = nltk.word_tokenize(doc_text)
            for token in tokens:
                if self.bag_of_words.get(token):
                    self.bag_of_words[token]['doc_ids'].append(doc_id)
                else:
                    self.bag_of_words[token] = {'doc_ids': [doc_id]}
        print(datetime.now().strftime("%H:%M:%S") + ": Prepared bag of words...")

        for key, value in self.bag_of_words.items():
            self.bag_of_words[key]['df'] = len(list(set(self.bag_of_words[key]['doc_ids'])))

    # Build the index (tf_idf_vector) from document dict and bag of words and build cache once index is built
    def build_index(self):
        documents_count = len(self.documents_dict.keys()) # number of ducments 445
        all_tokens = list(self.bag_of_words.keys()) # number of unique tokns 80000

        all_token_with_index = {}
        for token_index, token in enumerate(all_tokens):
            all_token_with_index[token] = token_index

        for doc_id, doc_text in self.documents_dict.items():
            doc_tokens = nltk.word_tokenize(doc_text)
            doc_tokens_counter = Counter(doc_tokens)
            doc_tokens_count = len(doc_tokens)

            np_token_array = np.zeros((len(all_tokens))) # 80000

            self.tf_idf_vector[doc_id] = {'tf_idf_vector':np_token_array}

            for token, value in self.bag_of_words.items():
                tf = 1 + np.log(doc_tokens_counter[token] + 1 /doc_tokens_count + 1)   # adding one due to infinite value so i can use lnc.ltc scheme
                #tf = doc_tokens_counter[token] / doc_tokens_count
                df = self.bag_of_words[token]['df']

                idf = np.log((documents_count + 1) / (df + 1))

                self.tf_idf_vector[doc_id]['tf_idf_vector'][all_token_with_index[token]] = tf*idf

            self.tf_idf_vector[doc_id]['tf_idf_vector'] = self.tf_idf_vector[doc_id]['tf_idf_vector'].tolist()

        self.cache_vector = self.tf_idf_vector
        with open('cache/cache_vector.json', 'w') as cache_vector_file:
            cache_vector_file.write(json.dumps(self.cache_vector))
        with open('cache/documents.json', 'w') as documents_file:
            documents_file.write(json.dumps(self.documents_dict))
        with open('cache/bag_of_words.json', 'w') as bag_of_words_file:
            bag_of_words_file.write(json.dumps(self.bag_of_words))
        with open('cache/doc_titles.json', 'w') as doc_titles_file:
            doc_titles_file.write(json.dumps(self.doc_titles))

    # Generate vector for query
    def generate_query_vector(self, tokens):
        documents_count = len(self.documents_dict.keys())
        all_tokens = list(self.bag_of_words.keys())

        all_token_with_index = {}
        for token_index, token in enumerate(all_tokens):
            all_token_with_index[token] = token_index

        query = np.zeros((len(all_tokens))) # [0,0,0,0,0,0]

        counter = Counter(tokens)
        words_count = len(tokens)

        for token in np.unique(tokens):

            tf = 1 + np.log(counter[token] + 1 / words_count + 1)
            #tf = counter[token] / words_count
            try:
                df = self.bag_of_words[token]['df']
            except:
                df = 0
            idf = np.log((documents_count + 1) / (df + 1))

            try:

                query[all_token_with_index[token]] = tf * idf
            except:
                pass
        return query

    # generate query vector using generate_query_vector method and then find cosine similarity with every document and print results
    def lookup_using_cosine_similarity(self, query):
        tokens = nltk.word_tokenize(query)

        print("\nQuery:", query)

        search_results = {}
        query_vector = self.generate_query_vector(tokens)  # 445 documents

        for doc_id, doc_value in self.cache_vector.items():
            cosine_similarity = 1 - spatial.distance.cosine(query_vector, np.asarray(doc_value['tf_idf_vector']))
            if cosine_similarity > 0:
                search_results[doc_id] = {"score": cosine_similarity, "title": self.doc_titles[doc_id]}

        items_to_return = 10
        if len(search_results) < 10:
            items_to_return = len(search_results)

        # Dictionary does not maintain order so commenting this and using list comprehension in next line
        # out = {key: value for key, value in sorted(search_results.items(), key=lambda item: item[1]['score'], reverse=True)[:items_to_return]}
        out = [value for key, value in
               sorted(search_results.items(), key=lambda item: item[1]['score'], reverse=True)[:items_to_return]]

        pp(out)











