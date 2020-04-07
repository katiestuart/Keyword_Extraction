##### Extract Key words from Text ########

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
import sys
sys.path.append("master_code/")
import urlscrape
import EntityExtract
import Vectoriser
import Utils
import importlib as im
from sklearn.cluster import KMeans
im.reload(urlscrape)
im.reload(EntityExtract)
im.reload(Vectoriser)
im.reload(Utils)
pd.set_option('display.max_rows', 500)

# current working directory
os.getcwd()
# find out what paths the code looks for functions in
# sys.path
# add a path to this list
# sys.path.append("master_code/")

HCP_data = pd.read_csv('HCP_Online_Behavior.csv')
HCP_data.head()

HCP_data['p_date'] = pd.to_datetime(HCP_data.day, infer_datetime_format=True)
HCP_data = HCP_data.drop(["total_impressions", "NPI"], axis = 1)
urls = list(HCP_data['contextualurl'])
# remove dupes
urls = list(dict.fromkeys(urls))
# urls[11]
##################################
# Pull back text from each link
##################################

output = []
failed = []
count = 0
for url in urls:
    url_dict, status_code = urlscrape.url_scrape(url)
    if status_code != 200:
        failed.append(url)
    else:
        output.append(url_dict)
    count += 1
    print(count)



# remove failed urls
output_df = pd.DataFrame(output)
output_df['url'] = urls
output_df = output_df[~output_df.url.isin(failed)].reset_index(drop=True)
output_df.to_csv("BS_output.csv")

output_df['p_text'][18]
urls[17]

##################################
# Pull keywords from Google API
##################################

keyword_results = EntityExtract.analyze_text_list(output_df)
keyword_results = keyword_results.rename(columns = {"name": "phrase"})

# Save file
# keyword_results.to_csv("api_output.csv")
# Import file
keyword_results = pd.read_csv("api_output.csv")
keyword_results = keyword_results.drop(["Unnamed: 0"], axis=1)



# keyword_results.phrase.groupby(keyword_results.doc_index).count()
# keyword_results[keyword_results.doc_index.values == 17]

# remove where type == Number
keyword_results = keyword_results[keyword_results.type.values != "NUMBER"]


####################################################
# Run TFIDF & Count Vectorizer on Key Entities
####################################################

# replace missing text with another col
output_df['p_text'] = np.where(output_df['p_text'] == '', output_df['all_text'], output_df['p_text'])

# create list of failed urls
failed_urls = [x for x in list(output_df["url"]) if x not in list(keyword_results["url"])]

# remove failed urls
output_df = output_df[~output_df.url.isin(failed_urls)].reset_index(drop=True)

# Pull clean text
ptext_array = list(output_df['p_text'])

# run countvec
df, vocab, doc_vectors = Vectoriser.Vec(ptext_array, "cv", min_df = 0, stop_words = 'english', ngram_range = (1,3))

# run tfidf
df_t, vocab_t, doc_vectors_t = Vectoriser.Vec(ptext_array, "tfidf", min_df = 0, stop_words = 'english', ngram_range = (1,3), norm = 'l2')

# Create final Table
df_t = df_t.rename(columns = {"count": "tfidf"})
vec_results = pd.merge(df, df_t, how= "left", on=['doc_index', 'phrase'])
# vec_results[vec_results.doc_index.values == 3]
vec_results = vec_results.drop(['doc_matrix_indices_x', 'doc_matrix_indices_y'], axis=1)

# Merge onto Keywords
keyword_results_n = pd.merge(keyword_results, vec_results, how= "left", on=['doc_index', 'phrase']).reset_index(drop=True)
keyword_results_n = keyword_results_n.drop(["wiki_url"], axis = 1)
keyword_results_n

# keyword_results_n['tfidf'].count()
# keyword_results_n['tfidf'].isna().count()

####################################################
# Select final keywords based on criteria
####################################################

# Change count to proportion
keyword_results_n = Utils.Count_prop(keyword_results_n)

# Scale TFIDF Scores
keyword_results_n = Utils.MinMax(keyword_results_n)

# Select final keywords based on criteria

# keyword_results_n.scaled_tfidf.groupby(keyword_results_n.doc_index).max()
#
# keyword_results_n["count_prop"].groupby(keyword_results_n.doc_index).max()

keyword_results_final = keyword_results_n[(keyword_results_n.scaled_tfidf > 0.05) & (keyword_results_n.count_prop > 0.005)]

# Review final Values
keyword_results_final[keyword_results_final.doc_index.values == 1].sort_values("scaled_tfidf", ascending = False).reset_index(drop=True)


####################################################
# Create Word Embeddings: BERT, Word2Vec, GloVe
####################################################

# Use Word2Vec

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

key_phrase = list(keyword_results_final.phrase)
key_phrase = list(dict.fromkeys(key_phrase))
print(len(key_phrase),len(set(key_phrase)))
# key_phrase

# import pre-trained model
model = gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)
# Train model
# model = Word2Vec([key_phrase], min_count=1, size = 100, window = 3, sg = 1)


# Get Embeddings for key phrases and remove phrases not in Word2Vec vocab from key phrase list

embeddings = []
key_phrase_y =[]
key_phrase_n =[]
in_vocab = 0
not_in_vocab = 0
for i, j in zip(key_phrase, range(len(key_phrase))):
    print("i", i)
    print("j", j)
    try:
        embeddings.append(model[i])
        key_phrase_y.append(i)
        in_vocab += 1
    except:
        #key_phrase.remove(key_phrase[j])
        key_phrase_n.append(i)
        not_in_vocab += 1

len(embeddings)

len(key_phrase_n)
len(key_phrase_y)

in_vocab
not_in_vocab

model.similarity('obstetrics', 'gynecology')
model.similarity('country', 'communities')

# Try BERT based on total doc

from sentence_transformers import SentenceTransformer

model_bert = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = model_bert.encode(ptext_array)


from sklearn.metrics.pairwise import cosine_similarity
# Country  vs. Communities
print(key_phrase[45], key_phrase[46])
cosine_similarity([sentence_embeddings[65]], [sentence_embeddings[66]])

cosine_similarity([sentence_embeddings[45]], [sentence_embeddings[46]])


#### Create cluster based on words and assign cluster to articles

clusters = Utils.cluster(25, embeddings, key_phrase_y)

cluster_names = []
for x in clusters.cluster_name.unique():
    #print('\ncluster:',x,'words:',len(clusters[clusters.cluster_name == x]),', '.join(clusters[clusters.cluster_name == x].phrase.values))
    cluster_names.append([x,len(clusters[clusters.cluster_name == x]),', '.join(clusters[clusters.cluster_name == x].phrase.values)])

# cluster_names_bert = pd.DataFrame(cluster_names, columns = ['cluster','num_words','cluster_words'])
# cluster_names_bert

cluster_names_df = pd.DataFrame(cluster_names, columns = ['cluster','num_words','cluster_words'])
cluster_names_df

# clusters[clusters.cluster_name == 6]

clust_count = pd.DataFrame(clusters["phrase"].groupby([clusters.cluster_name]).count()).reset_index()
# clust_count

# link back onto docs and figure out top CLUSTERS per doc

# merge on clusters by word
cluster_results = pd.merge(keyword_results_final, clusters, how= "left", on=['phrase'])

# cluster_results[cluster_results.doc_index.values == 0]

# work out proportion of cluster by doc
# divide by # words in each cluster for fair proportions
totals = pd.DataFrame(cluster_results["phrase"].groupby([cluster_results.doc_index, cluster_results.cluster_name]).count()).reset_index()
t = pd.DataFrame(totals["phrase"].groupby([totals.doc_index]).sum()).reset_index()
totals = pd.merge(totals, t, how= "left", on=['doc_index'])
totals["cluster_prop"] = totals.phrase_x/totals.phrase_y
totals = totals.drop(["phrase_y"], axis = 1)
totals = pd.merge(totals, clust_count, how= "left", on=['cluster_name'])
totals["cluster_prop_1"] = totals.cluster_prop/totals.phrase
totals[totals.doc_index.values == 3].sort_values("cluster_prop", ascending = False).reset_index()
totals = totals.drop(["phrase_x", "cluster_prop", "phrase"], axis = 1)

totals['cluster_name'] = totals['cluster_name'].astype('int')

totals = pd.merge(totals, cluster_names_df, left_on='cluster_name', right_on='cluster')

totals.sort_values(['doc_index','cluster_prop_1'], ascending=[1,0])


# Create single embedding for each article and cluster

urls = list(output_df['url'])

import matplotlib.pyplot as plt
clusters = Utils.cluster(10, sentence_embeddings)

urls[23]
clusters

#############################################################################################################################################################################################################
# links=[]
# for url in urls:
#     page = requests.get(url)
#
#     #print('Loaded page with: %s' % page)
#     soup = BeautifulSoup(page.content, 'html.parser')
#
#     for script in soup(["script", "style"]):
#         script.extract()
#
#     # soup.find_all('a')
#     text = soup.find_all(text=True)
#     len(text)
#
#     set([t.parent.name for t in text])
#
#     # Remove unwanted items from text
#     output = ''
#     blacklist = [
#     	'[document]',
#     	'noscript',
#     	'header',
#     	'html',
#     	'meta',
#     	'head',
#     	'input',
#     	'script',
#         'aside',
#         'button',
#         'address',
#         'footer',
#         'form',
#         'legend',
#         'nav',
#         'time',
#         'link',
#         'button',
#         'a',
#         'li',
#         'span'
#     	# there may be more elements you don't want, such as "style", etc.
#     ]
#
# for t in text:
#
# 	if t.parent.name not in blacklist:
# 		output += '{} '.format(t)
# 		print(t.parent.name ,t )
#
# output = output.replace("\n"," ")
#     #Clean text
#     # text = ' '.join(text)
#     output = output.replace("\n"," ")
#     text = ' '.join(text.split())
#
#     # r = requests.get(url)
#     # html = r.text
#     # soup = BeautifulSoup(html, 'lxml')
#     # Find heading
#     soup.find('h1')
#     heading = soup.find('h1').text.strip()
#     heading
#     for link in soup.find_all('a'):
#         print(link.get('href'))
