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
import urlscrape
import EntityExtract
import Vectoriser
import Utils
import sys

# current working directory
os.getcwd()
# find out what paths the code looks for functions in
# sys.path
# add a path to this list
sys.path.append("master_code/")

HCP_data = pd.read_csv('HCP_Online_Behavior.csv')

HCP_data.head()

HCP_data['p_date'] = pd.to_datetime(HCP_data.day, infer_datetime_format=True)
HCP_data = HCP_data.drop(["total_impressions", "NPI"], axis = 1)
urls = list(HCP_data['contextualurl'])
# remove dupes
urls = list(dict.fromkeys(urls))

##################################
# Pull back text from each link
##################################

output = []
for url in urls[0:5]:
    url_dict = urlscrape.url_scrape(url)
    output.append(url_dict)
output_df = pd.DataFrame(output)
output_df['url'] = urls[0:5]
output_df.head()

# output_df['all_text'][2]
# print(output_df['iframes'][4])
# urls[1]

##################################
# Pull keywords from Google API
##################################

keyword_results = EntityExtract.analyze_text_list(output_df)
keyword_results = keyword_results.rename(columns = {"name": "phrase"})

# Save file
keyword_results.to_csv("api_output.csv")

keyword_results.head()


####################################################
# Run TFIDF & Count Vectorizer on Key Entities
####################################################

# Pull clean text
ptext_array = list(output_df['p_text'])
# run countvec
df, vocab, doc_vectors = Vectoriser.Vec(ptext_array, "cv", min_df = 0, max_df = 0.95, stop_words = 'english', ngram_range = (1,3))
# run tfidf
df_t, vocab_t, doc_vectors_t = Vectoriser.Vec(ptext_array, "tfidf", min_df = 0, max_df = 0.95, stop_words = 'english', ngram_range = (1,3), norm = 'l2')

# Create final Table
df_t = df_t.drop(['phrase'], axis=1)
df_t = df_t.rename(columns = {"count": "tfidf"})
vec_results = pd.merge(df, df_t, how= "left", on=['doc_index', 'doc_matrix_indices'])
vec_results.head()

# Merge onto Keywords

keyword_results_n = pd.merge(keyword_results, vec_results, how= "left", on=['doc_index', 'phrase'])

####################################################
# Select final keywords based on criteria
####################################################

# Change count to proportion
keyword_results_n = Utils.Count_prop(keyword_results_n)

# Scale TFIDF Scores
keyword_results_n = Utils.MinMax(keyword_results_n)

# Select final keywords based on criteria
pd.set_option('display.max_rows', 500)
# keyword_results_n[keyword_results_n.count_prop>0].sort_values('count_prop', ascending=True)
# keyword_results_n[keyword_results_n.doc_index.values == 3]
# keyword_results_n = keyword_results_n.drop(['wiki_url'], axis=1)

keyword_results_n.scaled_tfidf.groupby(keyword_results_n.doc_index).max()

keyword_results_n["count_prop"].groupby(keyword_results_n.doc_index).max()

keyword_results_final = keyword_results_n[(keyword_results_n.scaled_tfidf > 0.05) & (keyword_results_n.count_prop > 0.005)]

# Review final Values
keyword_results_final[keyword_results_final.doc_index.values == 3].sort_values("scaled_tfidf", ascending = False).reset_index()


#### Create cluster based on words and assign cluster to articles






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
