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

HCP_data = pd.read_csv('HCP_Online_Behavior.csv')

HCP_data.head()

HCP_data['p_date'] = pd.to_datetime(HCP_data.day, infer_datetime_format=True)
HCP_data = HCP_data.drop(["total_impressions", "NPI"], axis = 1)


##################################
# Pull back text from each link
##################################

urls = list(HCP_data['contextualurl'])
urls = list(dict.fromkeys(urls))

def extract_meta_tags(soup):

    meta_tags = []

    # All meta tags are divided into these four tags
    meta_names = ['property', 'name', 'http-equiv', 'charset']

    # Try find all meta tags contained
    for i in range(len(soup.find_all("meta"))):
        for idx in meta_names:

            try:
                meta_tags.append(soup.find_all("meta")[i][idx])
            except:
                pass

    meta_values = []

    # Retreive all values under each meta tag that exists
    for tag in meta_tags:
        for idx in meta_names:
            try:
                desc = soup.findAll(attrs={idx: re.compile(tag, re.I)})
                meta_values.append(desc[0]['content'].encode('utf-8'))

            except:
                pass

    # Return dictionary of meta tags and their respective values
    meta_dict = dict(zip(meta_tags,meta_values))
    return meta_dict


def url_scrape(url):
    """
        Input a URL and output a dictionary containing all common HTML features
    """
    global url_dict
    # import urllib.request
    ## Remove unnecessary characters -- whitespace character
    url = re.sub(r'\s+', '', url)
    ## Create empty dictionary with url as key
    url_dict = {}
    ## bs4
    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}
    r = requests.get(url, headers=headers, timeout=10)
    html = None
    links = None
    # if it's loaded the page properly
    if r.status_code == 200:
        html = r.text
        soup = BeautifulSoup(html, 'lxml')
        for script in soup(["script", "style"]):
            script.extract()

    ## Extract all features
    url_dict['p_text'] = "".join([element.text for element in soup.findAll('p')])
    url_dict['h1_text'] = "".join([element.text for element in soup.findAll('h1')])
    # url_dict['h1_text'] = url_dict['h1_text'].replace("\n"," ")
    # url_dict['h1_text'] = url_dict['h1_text'].replace("\r"," ")
    url_dict['hrefs'] = [link.get('href') for link in soup.findAll('a')]
    url_dict['img_links'] = [image['src'] for image in soup.findAll('img')]
    url_dict['iframes']= soup.find_all('iframe')
    # url_dict['videos']= soup.find_all('video')
    try:
        url_dict['videos']= re.search("(?P<url>https?://[^\s]+)", str(url_dict['iframes'])).group("url").replace('"',"")
    except:
        url_dict['videos'] = ''
    meta = extract_meta_tags(soup)
    url_dict['meta'] = meta
    try:
        url_dict['keywords'] = meta['keywords']
    except:
        url_dict['keywords'] = ''

    for i in ['p_text', 'h1_text', 'keywords']:

        url_dict[i] = str(url_dict[i]).replace("\n"," ")
        url_dict[i] = str(url_dict[i]).replace("\r"," ")
        url_dict[i] = str(url_dict[i]).replace("\t"," ")

    # url_dict['html'] = html
    if soup.title:
        url_dict['title'] = soup.title.string
    else:
        url_dict['title'] = ''
    ## add full text
    for script in soup(["script", "style"]):
        script.extract()
    org_text = soup.get_text()
    text = org_text.replace("\n"," ")
    text = ' '.join(text.split())
    url_dict['all_text'] = text
    return url_dict


output = []
for url in urls[0:5]:
    url_dict = url_scrape(url)
    output.append(url_dict)
output_df = pd.DataFrame(output)
output_df['url'] = urls[0:5]
output_df.head()

output_df['all_text'][2]
print(output_df['iframes'][4])
urls[1]

ptext_array = list(output_df['p_text'])


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


response
def analyze_text_entities(text):
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

    full_keyword_DF = pd.DataFrame()

    counter = 0
    for i in range(len(docs)):
        try:
            keyword_DF = analyze_text_entities(output_df[i:i+1]["p_text"].values[0])
        except:
            keyword_DF = analyze_text_entities(output_df[i:i+1]["all_text"].values[0])

        keyword_DF["doc_index"] = counter
        full_keyword_DF = full_keyword_DF.append(keyword_DF)
        counter += 1

    return full_keyword_DF


keyword_results = analyze_text_list(output_df)
keyword_results = test.rename(columns = {"name": "phrase"})

keyword_results.to_csv("api_output.csv")

# for i in range(0, len(output_df["p_text"])):
#     print(i)
#     print(output_df[i]["url"])
#
# output_ptext = list(output_df["p_text"])
# output_urls = list(output_df["urls"])

output_df[0:1]["url"]
output_df['p_text'][1]
text = output_df['p_text'][1]
keyword_results = analyze_text_entities(text)

keyword_results[keyword_results.doc_index.values == 1]



## Run through a countvectoriser to pull top entities
#
# vectorizer = CountVectorizer(lowercase=True, stop_words=None, ngram_range=(1, 3), min_df=0.25)
# vectorizer.fit(ptext_array[1:5])
#
# vector = vectorizer.transform(ptext_array[1:5])
# vectorizer.vocabulary_
#
# dfvec = pd.DataFrame({'doc_index':vector.nonzero()[0],
#                    'doc_matrix_indices':vector.nonzero()[1], # index of word
#                    'count':vector.data})
# dfvec['phrase']=[vocab[x] for x in dfvec.doc_matrix_indices]
#
# vocab = vectorizer.get_feature_names()
#
# vector[0].toarray()[0]





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

# run countvec
df, vocab, doc_vectors = Vec(ptext_array, "cv", min_df = 0, max_df = 0.95, stop_words = 'english', ngram_range = (1,3))
# run tfidf
df_t, vocab_t, doc_vectors_t = Vec(ptext_array, "tfidf", min_df = 0, max_df = 0.95, stop_words = 'english', ngram_range = (1,3), norm = 'l2')
df_t = df_t.drop(['phrase'], axis=1)
df_t = df_t.rename(columns = {"count": "tfidf"})

vec_results = pd.merge(df, df_t, how= "left", on=['doc_index', 'doc_matrix_indices'])

vec_results.head()

# Add on count countvectoriser

keyword_results_n = pd.merge(keyword_results, vec_results, how= "left", on=['doc_index', 'phrase'])
# Change count to proportion

# sum counts by doc index
doc_counts = pd.DataFrame(keyword_results_n["count"].groupby(keyword_results_n.doc_index).sum()).reset_index()
keyword_results_n = pd.merge(keyword_results_n, doc_counts, how= "left", on=['doc_index'])
keyword_results_n["count_prop"] = keyword_results_n.count_x/keyword_results_n.count_y
keyword_results_n = keyword_results_n.drop(['count_y'], axis=1)

keyword_results_n = keyword_results_n.fillna(0)
tfidf = np.array(keyword_results_n["tfidf"])
tfidf = tfidf.reshape(-1,1)
tfidf.shape


scaler = MinMaxScaler()
tfidf = scaler.fit_transform(tfidf)
keyword_results_n["scaled_tfidf"] = tfidf

pd.set_option('display.max_rows', 500)
keyword_results_n[keyword_results_n.count_prop>0].sort_values('count_prop', ascending=True)
keyword_results_n[keyword_results_n.doc_index.values == 3]
keyword_results_n = keyword_results_n.drop(['wiki_url'], axis=1)

# Select final keywords based on criteria

keyword_results_final = keyword_results_n[(keyword_results_n.scaled_tfidf > 0.05) & (keyword_results_n.count_prop > 0.005)]

keyword_results_final[keyword_results_final.doc_index.values == 3].sort_values("scaled_tfidf", ascending = False).reset_index()

keyword_results_n.scaled_tfidf.groupby(keyword_results_n.doc_index).max()

keyword_results_n["count_prop"].groupby(keyword_results_n.doc_index).max()


#### Create cluster based on words and assign cluster to articles


###########################################################################################
# Google Classification
###########################################################################################
def classify_text(text):
    client = language.LanguageServiceClient()
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    response = client.classify_text(document=document)

    # for category in response.categories:
    #     print('=' * 79)
    #     print('category  : {}'.format(category.name))
    #     print('confidence: {:.0%}'.format(category.confidence))

    return response

categories = classify_text(text)

classify_text(output_df['p_text'][3])



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



# def run_multi_sitescrape(df,
#                          save_path = 'data/',
#                          verbose=1):
#
#
#     """
#         Run mutliple sitescrapes in a list, storing after each site
#
#         Datafiles used
#             {save_path}all_links/all_links_{save_name}.csv
#     """
#
#     from datetime import datetime
#     if not os.path.exists(save_path+'/htmlscrape'):
#         os.makedirs(save_path+'/htmlscrape')
#     counter=0
#     if verbose>0:
#         print('''
#                 ------------------------
#                 Full Folder Structure
#                 ------------------------
#                 - data
#                     - htmlscrape
#                         - html_full_data{save_name}.csv
#                 ''')
#     # for s in sitelist:
#     #     save_name = s.replace('https://','').replace('.','_').replace('/','')
#         # df = pd.read_csv(save_path+'all_links/all_links_'+save_name+'.csv',encoding='utf-8')
#         # if verbose>0:
#         #     print('SITE:',counter)
#         #     print(save_name,len(df))
#         start_time = datetime.now()
#         df['html_features'] = df['url'].apply(lambda x: try_except_url_scrape(x))
#         time_elapsed = datetime.now() - start_time
#         if verbose>0:
#             print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
#             print('*'*50)
#         df.to_csv(save_path+'htmlscrape/html_full_data'+counter+'.csv',encoding='utf-8',index=None)
#         counter+=1
#



# url_scrape(urls[0])
# df = pd.DataFrame(urls[0:5], columns = ['url'])
# df['html_features'] = df['url'].apply(lambda x: try_except_url_scrape(x))
#
# #convert to dict (Python 3 stores as str)
# df['html_features_new'] = df['html_features'].map(lambda d : row2dict(d))
# df['html_features_new'].apply(pd.Series)
# df = pd.concat([df.drop(['html_features_new'], axis=1), df['html_features_new'].apply(pd.Series)], axis=1)
#
# df['p_text'] = url_dict[0]['p_text']
#
#
# url_dict
#
# import ast
# ast.literal_eval(df['html_features'][1]['p_text'])
#
# run_multi_sitescrape(urls[0:5])
