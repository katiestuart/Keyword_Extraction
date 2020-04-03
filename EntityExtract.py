import pandas as pd
pd.options.display.max_columns = 999
from bs4 import BeautifulSoup
import requests
import urllib.parse as urlparse_lib
import re
import time
import os
import pickle
from time import sleep as wait
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import urlscrape
import sys

##################################
# Pull keywords from Google API
##################################
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


# To help set up google cloud api
# https://cloud.google.com/natural-language/docs/quickstart#quickstart-analyze-entities-cli

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/katiestuart/Keyword-Extraction-2804f152e6bc.json"

def implicit():
    from google.cloud import storage

    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    storage_client = storage.Client()

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)


def analyze_text_entities(text):

    """
        Takes in string & output key entities from Google API
    """

    global response
    client = language.LanguageServiceClient()

    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    response = client.analyze_entities(document=document)

    output = []
    for entity in response.entities:
        # print('=' * 79)
        results = [
            (entity.name),
            (enums.Entity.Type(entity.type).name),
            (entity.salience),
            (entity.metadata.get('wikipedia_url', '-')),
            # (entity.metadata.get('mid', '-')),
        ]

        output.append(results)
        # for k, v in results:
        #     print('{:15}: {}'.format(k, v))
    keyword_DF = pd.DataFrame(output)
    keyword_DF = keyword_DF.rename(columns = {0:"name", 1:"type", 2:"salience", 3:"wiki_url"})
    keyword_DF = keyword_DF.drop_duplicates(subset="name", keep = "first")

    return keyword_DF

def analyze_text_list(docs):

    """
        Loops around text using text column from DF and pulls back entities iteratively
    """

    full_keyword_DF = pd.DataFrame()

    counter = 0
    for i in range(len(docs)):
        try:
            keyword_DF = analyze_text_entities(docs[i:i+1]["p_text"].values[0])
        except:
            keyword_DF = analyze_text_entities(docs[i:i+1]["all_text"].values[0])

        keyword_DF["doc_index"] = counter
        full_keyword_DF = full_keyword_DF.append(keyword_DF)
        counter += 1

    return full_keyword_DF
