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

##################################
# Pull back text from each link
##################################

def extract_meta_tags(soup):

    """
        Takes in a soup from BS4 and produces a dictionary containing all meta tags on a page
    """

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
    # headers = {
    #     'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}

    r = requests.get(url, headers=headers, timeout=20)
    html = None
    links = None
    # if it's loaded the page properly

    if r.status_code == 200:
        html = r.text
        soup = BeautifulSoup(html, 'lxml')
        for script in soup(["script", "style"]):
            script.extract()
    else:
        print(r.status_code)


    ## Extract all features
    try:
        url_dict['p_text'] = "".join([element.text for element in soup.findAll('p')])
    except:
        url_dict['p_text'] = ''
    try:
        url_dict['h1_text'] = "".join([element.text for element in soup.findAll('h1')])
    except:
        url_dict['h1_text'] = ''
    url_dict['hrefs'] = [link.get('href') for link in soup.findAll('a')]
    try:
        url_dict['img_links'] = [image['src'] for image in soup.findAll('img')]
    except:
        url_dict['img_links'] = ''
    try:
        url_dict['iframes'] = soup.find_all('iframe')
    except:
        url_dict['iframes'] = ''

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
    return url_dict, r.status_code
