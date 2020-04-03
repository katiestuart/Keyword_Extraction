####################################################
# Run TFIDF & Count Vectorizer on Key Entities
####################################################

import pandas as pd
pd.options.display.max_columns = 999
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def Vec(data, vectorizer = "cv", min_df = 5, max_df = 0.95, max_features = 8000, stop_words = 'english', ngram_range = (1,3), norm = None):

    ## Run CountVectorizer
    if vectorizer == "cv":
        vec = CountVectorizer(
                                min_df = min_df, # min document frequency: number of times word has to be in all documents to be included
                                max_df = max_df, # max document frequency: to remove words like a, the, and - that appear all the time
                                # max_features = max_features, # max number of words to include in vectorizer
                                stop_words = stop_words, #take out stop words
                                ngram_range= ngram_range, # (1,1) unigrams only, (1,2) both uni and bi-grams, (2,2) bigrams only
                                lowercase = False
                                )

    ## Run TFIDF
    if vectorizer == "tfidf":
        vec = TfidfVectorizer(
                                min_df = min_df, # min document frequency: number of times word has to be in all documents to be included
                                max_df = max_df, # max document frequency: to remove words like a, the, and - that appear all the time
                                max_features = max_features, # max number of words to include in vectorizer
                                stop_words = stop_words, #take out stop words
                                ngram_range= ngram_range, # (1,1) unigrams only, (1,2) both uni and bi-grams, (2,2) bigrams only
                                norm = norm,
                                lowercase = False
                                )

    vec.fit(data)
    # returns sparse matrix: doc index x word index
    doc_vectors = vec.transform(data)
    # Get the vocab, words in the doc vectors
    vocab = vec.get_feature_names()

    # Create a dataframe of all docs, their words and TFIDF scores
    df = pd.DataFrame({'doc_index':doc_vectors.nonzero()[0],
                       'doc_matrix_indices':doc_vectors.nonzero()[1], # index of word
                       'count':doc_vectors.data})

    # Add the actual word from the vocab
    df['phrase']=[vocab[x] for x in df.doc_matrix_indices]

    # Add rank based on tfidf for each document
    df = df.sort_values(['doc_index','count'],ascending=[1,0])
    # df['rank'] = df.groupby('doc_index')['count'].rank(ascending=False)

    return df, vocab, doc_vectors
