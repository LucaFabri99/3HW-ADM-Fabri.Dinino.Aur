import pandas as pd
import math
import matplotlib.pyplot as plt
import lxml
import numpy as np

import os                                                      #Needed to move between OS folders

import re
### Import libraries

from bs4 import BeautifulSoup                                  #Scraper 
import requests                                                #URL drainer

from tqdm import tqdm

from datetime import datetime                                  #To be leveraged to define datetime objects  

from collections import Counter
from functools import reduce

import nltk                                                    #Text processing library
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


import locationtagger
import pandas as pd
import plotly.express as px
import dash
from dash import dcc 

from sklearn.feature_extraction.text import TfidfVectorizer    #Useful already implemented tfidf vectorizer from scikit learn library
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import heapq                                                   #Library with useful methods to define heap data structures

import spacy
from itertools import combinations


###_____________________SEARCH ENGINE USEFUL METHODS________________________________________________

def ntlk_analysis(text):
    '''
    This function cleans the text in input, removing puntactions and the most common English words.
    It returns a list of words reduced to its stem or root format.
    '''
    final_words = []
    tokens = word_tokenize(text.lower()) # tokenize the text 
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer() # stemmer function
    for token in tokens:
        if token not in stop_words and token.encode().isalpha(): # remove puntactions and 
                                                                # most common English words
            stemming_token = ps.stem(token) # reduce words to its root format 
                                            # through stemmer function
            final_words.append(stemming_token)
            
    return final_words

def open_vocabulary(filename):
    '''
    This function receives as input the name of a text file and
    converts its content into a dictionary type object.
    '''
    vocabulary = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            word = line.split(':')[0]
            term_id = line.split(':')[1]
            vocabulary[word] = int(term_id)
     
    return vocabulary
            
def open_inverted_index(filename):
    '''
    This function receives as input the name of a text file and
    converts its content into a dictionary type object.
    '''
    inverted_index = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            word = int(line.split(':')[0])
            indexing = line.split(':')[1].strip('\n')
            indexing = indexing.split(',')
            if indexing[-1] == ' ':
                indexing = indexing[:-1]
            inverted_index[word] = list(map(int, [string.strip() for string in indexing]))
            
    return inverted_index

def open_inverted_tfidf_index(filename):
    '''
    This function receives as input the name of a text file and
    converts its content into a dictionary type object.
    '''
    inverted_index = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            word = int(line.split(':')[0])
            indexing = line.split(':')[1].strip('\n')
            indexing = indexing.split(',')
            if indexing[-1] == ' ':
                indexing = indexing[:-1]
            inverted_index[word] = indexing       
    return inverted_index

def search_match(query, df, vocabulary, inverted_index):
    '''
    This function finds the common documents in the dataframe which contain 
    the terms of the user's query.
    It returns the list of documents indexes, the result of the intersection.
    '''
    query = ntlk_analysis(query)  # pre-process user's query
    try:
        match = {vocabulary[term]: [] for term in query} # return the corresponding id of those terms
    except KeyError:
        return None

    for key,list_of_values in inverted_index.items():
        if key in match:
            for value in list_of_values:
                if value not in match[key]:
                    match[key].append(value) # find the documents indexes 
                                            # which contain the terms of the query 
                    
    final_values = list(match.values())
    intersection = set.intersection(*map(set,final_values)) # find the common indexes documents
    
    return intersection


def visualize_result(df, array):
    '''
    This function allows the visualization of some specific indexes and columns in the dataframe.
    '''
    result = []
    for index in array:
        series = df[['placeName', 'placeDesc','placeURL']].loc[index]
        result.append(series)
    
    return pd.DataFrame(result)


### 2.2. Conjunctive query & Ranking score

def heap_top_k(k, array, reverse):
    '''
    This functions converts an array type object into a heapify structure 
    and returns the top-k elements in the array. 
    The "reverse" parameter is a boolean which indicates how the sorting should be done.
    '''
    heapq.heapify(array)  # heapify structure
    if reverse:
        top_k = (heapq.nlargest(k, array, lambda x:-x[1])) # ascending order
    else:
        top_k = (heapq.nlargest(k, array, lambda x:x[1])) # descending order
    return top_k

def cosine_similarity(v1,v2):
    '''
    This function simply computes the cosine similarity from its definition.
    '''
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


def tfidf_search_match(query, df, vocabulary, tfidf_inverted_index, idf, k):
    '''
    This function finds the score of Search Engine based on tfidf value and cosine similarity. 
    It returns the top-k documents in the dataframe sorted by the highest score.
    '''
    result = []
    query = ntlk_analysis(query)
    query_tfidf = {vocabulary[term]: Counter(query)[term]*idf[term]/len(query) for term in query}
    match = {vocabulary[term]: [] for term in query}

    for key, list_of_tuples in tfidf_inverted_index.items():
        if key in match.keys():
            for tuple in list_of_tuples:
                match[key].append(tuple[0])
    
    intersec = set.intersection(*map(set,list(match.values())))
    
    k_nearest = []
    for index in intersec:
        vec = []
        for id in match.keys():
            for tuple in tfidf_inverted_index[id]:
                if tuple[0] == index:
                    vec.append(tuple[1])
        k_nearest.append((index, cosine_similarity(vec, list(query_tfidf.values()))))

        
    k_nearest = heap_top_k(k, k_nearest, False) # find top-k documents
    
    for idx,score in k_nearest:
        series = df[['placeName', 'placeDesc','placeURL']].loc[idx]
        series['cosineSimilarity'] = score
        result.append(series)
    
    return pd.DataFrame(result)


### 3. Define a new score!

def jaccard(list1, list2):
    '''
    This function simply computes the Jaccard similarity from its definition.
    '''
    return len(set(list1).intersection(list2))/len(set(list1).union(list2))


def new_score_jaccard(query, df, vocabulary, inverted_index, k):
    '''
    This function finds a new score of Search Engine based on Jaccard similarity. 
    It returns the top-k documents in the dataframe sorted by the highest score.
    '''
    intersection = search_match(query, df, vocabulary, inverted_index)
    query = ntlk_analysis(query)
    rank = []

    for idx in intersection:
        if type(df.loc[idx]['placeTags']) == str:
            tags = ntlk_analysis(df.loc[idx]['placeTags'])
            jaccard_idx = jaccard(tags, query) # compute Jaccard similarity
            rank.append((idx,jaccard_idx))
        else:
            rank.append((idx, -1))
            
    rank = heap_top_k(k, rank, False) # find top-k documents 
    result = []
    for idx,score in rank:
        series = df[['placeName', 'placeDesc','placeURL']].loc[idx]
        series['newScore'] = score
        result.append(series)
    
    return pd.DataFrame(result)

def evaluation(dataframe, query, relevant_word, vocabulary, similarityScore, inverted_index, tfidf_inverted_index, idf):
    matched = search_match(query, dataframe, vocabulary, inverted_index)
    k = len(matched)
    matched_tfidf = tfidf_search_match(query, dataframe, vocabulary, tfidf_inverted_index, idf, k)
    matched_jaccard = new_score_jaccard(query, dataframe, vocabulary, inverted_index, k)

    results = matched_tfidf.merge(matched_jaccard)

    relevant_retrieved = dataframe[dataframe.placeName.str.contains(relevant_word)]

    relevant_index = np.zeros(k)
    for ind in results.index:
        if ind in relevant_retrieved.index:
            relevant_index[ind] = 1

    results['relevance'] = relevant_index

    thresholds = np.linspace(0,1,21)
    TP = []
    TN = []
    FP = []
    FN = []

    Accuracy = []
    Precision = []
    Recall = []
    F = []
    for t in np.flip(thresholds):
        predictions = np.array([0 if y < t else 1 for y in results.similarity_score])
        matrix = confusion_matrix(predictions, relevant_index)

        TP.append(matrix[1,1])
        TN.append(matrix[0,0])
        FP.append(matrix[0,1])
        FN.append(matrix[1,0])

        Accuracy.append(metrics.accuracy_score(relevant_index, predictions))
        Precision.append(metrics.precision_score(relevant_index, predictions))
        Recall.append(metrics.recall_score(relevant_index, predictions))
        F.append(metrics.f1_score(relevant_index, predictions))

    metr = pd.DataFrame({
    'Threshold': np.flip(thresholds),
    'TP': TP,
    'TN': TN,
    'FP': FP,
    'FN': FN,
    'Accuracy': np.array(Accuracy),
    'Precision': np.array(Precision),
    'Recall': np.array(Recall),
    'F1': np.array(F)
    })

    norm_tp = np.dot(metr['TP'], 1/max(metr['FN']))
    norm_fp = np.dot(metr['FP'], 1/max(metr['TN']))

    return metrics.auc(norm_fp, norm_tp)


