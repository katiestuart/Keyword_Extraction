import pandas as pd
pd.options.display.max_columns = 999
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def Count_prop(data):

    # sum counts by doc index
    doc_counts = pd.DataFrame(data["count"].groupby(data.doc_index).sum()).reset_index()
    data = pd.merge(data, doc_counts, how= "left", on=['doc_index'])
    data["count_prop"] = data.count_x/data.count_y
    data = data.drop(['count_y'], axis=1)

    return data

def MinMax(data):

    # Remove nulls & Shape Data
    data = data.fillna(0)
    tfidf = np.array(data["tfidf"])
    tfidf = tfidf.reshape(-1,1)

    # Run scaler
    scaler = MinMaxScaler()
    tfidf = scaler.fit_transform(tfidf)
    data["scaled_tfidf"] = tfidf

    return data
