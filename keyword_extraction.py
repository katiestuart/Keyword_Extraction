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


output_df['videos'][4]
print(output_df['iframes'][4])
urls[4]


##################################
# Pull keywords from Google API
##################################


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
