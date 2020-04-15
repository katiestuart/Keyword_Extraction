import pandas as pd
pd.options.display.max_columns = 999
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def Count_prop(data):

    """
        Works out proportion word appears based on total # of words in document
    """

    # sum counts by doc index
    doc_counts = pd.DataFrame(data["count"].groupby(data.doc_index).sum()).reset_index()
    data = pd.merge(data, doc_counts, how= "left", on=['doc_index'])
    data["count_prop"] = data.count_x/data.count_y
    data = data.drop(['count_y'], axis=1)

    return data

def MinMax(data):

    """
        Scales TFIDF so Criteria can be applied to select final keywords
    """

    # Remove nulls & Shape Data
    data = data.fillna(0)
    tfidf = np.array(data["tfidf"])
    tfidf = tfidf.reshape(-1,1)

    # Run scaler
    scaler = MinMaxScaler()
    tfidf = scaler.fit_transform(tfidf)
    data["scaled_tfidf"] = tfidf

    return data


def cluster(cluster_type, n_clusters, sentence_embeddings, key_phrase = None, create_plot = 'Y'):

    if key_phrase == None:
        key_phrase = list(range(0,len(sentence_embeddings),1))

    # Run KMeans
    if cluster_type == 0:
        # CREATE CLUSTERS
        kmeans = KMeans(n_clusters= n_clusters)
        # fit kmeans object to data
        kmeans.fit(sentence_embeddings)
        # print location of clusters learned by kmeans object
    #     print(kmeans.cluster_centers_)
        # save new clusters for chart
        y_km = kmeans.fit_predict(sentence_embeddings)

    # Run Hierarchical Clustering
    if cluster_type == 1:
        from sklearn.cluster import AgglomerativeClustering
        cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        y_km = cluster.fit_predict(sentence_embeddings)

    se_array = np.array(sentence_embeddings)

    # CREATE PLOT
    fulldf=pd.DataFrame()
    for i in range(0, n_clusters):
    #     print(i)
        if create_plot == 'Y':
            plt.scatter(se_array[y_km ==i,0][0], se_array[y_km == i,1][0], s=100)
        tempdf = pd.DataFrame(np.array(key_phrase)[[y_km == i,0][0]])
        tempdf.columns=['phrase']
        tempdf['cluster_name']=i
        fulldf = fulldf.append(tempdf)
    if create_plot == 'Y':
        plt.legend(range(0 , n_clusters), loc='center left', bbox_to_anchor=(1, 0.5))

    return fulldf
