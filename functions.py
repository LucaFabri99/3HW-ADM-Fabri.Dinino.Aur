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

import heapq                                                   #Library with useful methods to define heap data structures

import spacy
from itertools import combinations

### 1.3 Parse downloaded pages

def string_to_datetime(string):
    '''
     This function converts an object of type string to one of type datetime.
    '''
    return str(datetime.strptime(string, '%B %d, %Y').date())

def darkAtlasScraper(text, filename):
    '''
    '''
    soup = BeautifulSoup(text)
    
    scraped = {'placeName': 'NaN',
               'placeTags': 'NaN',
               'numPeopleVisited': 'NaN',
               'numPeopleWant': 'NaN',
               'placeDesc': 'NaN',
               'placeShortDesc':'Nan',
               'placeNearby': 'NaN',
               'placeAddress': 'NaN',
               'placeAlt': 'NaN',
               'placeLong': 'NaN',
               'placeEditors': 'NaN',
               'placePubDate': 'NaN',
               'placeRelatedPlaces': 'NaN',
               'placeRelatedLists': 'NaN',
               'placeURL': 'NaN'}          
    
    try:
        scraped['placeName'] = soup.find_all('h1',{'class':'DDPage__header-title'})[0].contents[0]
    except IndexError:
        pass
           
    try:
        scraped['placeTags'] = list(map(lambda s:s.strip(),
                                        [tag.contents[0] for tag in soup.find_all('a',{'class':'itemTags__link js-item-tags-link'})]))
    except IndexError:
        pass
    
    
    counters = soup.find_all('div',{'class':'title-md item-action-count'})
    try:
        scraped['numPeopleVisited'] = int(counters[0].contents[0])
    except IndexError:
        pass
    try:
        scraped['numPeopleWant'] = int(counters[1].contents[0])
    except IndexError:
        pass
    

    place_desc = ''
    for paragraph in soup.find_all('div',{'class':'DDP__body-copy'})[0].find_all('p'):
        for element in paragraph.contents:
            if re.search('<[^>]*>', str(element)):
                element = re.sub('<[^>]*>', "", str(element))
                place_desc += element
            else:
                place_desc += str(element)
    scraped['placeDesc'] = place_desc
    
    try:
        scraped['placeShortDesc'] = soup.find_all('h3',{'class':'DDPage__header-dek'})[0].contents[0].replace(u'\xa0', u'')
    except IndexError:
        pass

    nearby = []
    try:
        for nearbies in soup.find_all('div',{'class':'DDPageSiderailRecirc__item-text'}):
            nearby.append(nearbies.find_all('div',{'class':'DDPageSiderailRecirc__item-title'})[0].contents[0])
        scraped['placeNearby'] = nearby
    except IndexError:
        pass
    
    try:
        address = (str(soup.find_all('aside',{'class':'DDPageSiderail__details'})[0]
                           .find_all('address',{'class':'DDPageSiderail__address'})[0]
                           .find_all('div')[0])
                           .split('\n', 1)[0])
        scraped['placeAddress'] = re.sub('<[^>]*>', " ", address)
    except IndexError:
        pass
    
    coordinates = soup.find_all('div',{'class':'DDPageSiderail__coordinates js-copy-coordinates'})[0].contents[2]
    scraped['placeAlt'] = float(coordinates.split()[0][:-1])
    scraped['placeLong'] = float(coordinates.split()[1])


    editorsoup = soup.find_all('a',{'class':'DDPContributorsList__contributor'})
    scraped['placeEditors'] = [stuff.find_all('span')[0].contents[0] 
                               for stuff in editorsoup 
                               if len(stuff.find_all('span')) > 0]
    if not scraped['placeEditors']:
        zzz = soup.find_all('div',{'class':'ugc-editors'})
        flag = 0
        for soupper in zzz:
            if soupper.find_all('h6')[0].contents[0] == 'Added by':
                flag = 1
                break
        try:
            editorsoup = soup.find_all('div',{'class':'ugc-editors'})[flag].find_all('a',{'class':'DDPContributorsList__contributor'})
            scraped['placeEditors'] = [editors.contents[0]
                                       for editors in editorsoup]
        except IndexError:
            pass
            
    try:
        scraped['placePubDate'] = string_to_datetime(soup.find_all('div',{'class':'DDPContributor__name'})[0].contents[0])
    except IndexError:
        pass

    kircher = soup.find_all('div',{'class':'athanasius'})
    for piece in kircher:
        for piecer in piece.find_all('div',{'class':'CardRecircSection__title'}):
            if piecer.contents[0] == 'Related Places':
                scraped['placeRelatedPlaces'] = [re.sub('<[^>]*>', "", str(chunk.contents[1])) 
                                                 for chunk in piece.find_all('h3',{'class':'Card__heading --content-card-v2-title js-title-content'})]
            elif 'Appears in' in piecer.contents[0]:
                scraped['placeRelatedLists'] =  [re.sub('<[^>]*>', "", str(chunk.contents[1])) 
                                                 for chunk in piece.find_all('h3',{'class':'Card__heading --content-card-v2-title js-title-content'})]
    
    scraped['placeURL'] = 'https://www.atlasobscura.com/places/' + filename[:-4]
    
    return scraped


def open_dataset(filename):
    '''
    This functions converts a CSV file into a dataframe and initializes its columns.
    '''
    df = pd.read_csv(filename)
    df = df.iloc[:, :-2]
    df.columns = ['placeName', 'placeTags', 'numPeopleVisited', 'numPeopleWant', 'placeDesc',
                                 'placeShortDesc', 'placeNearby', 'placeAddress', 'placeAlt', 'placeLong', 'placeEditors', 
                                 'placePubDate', 'placeRelatedPlaces', 'placeRelatedLists', 'placeURL']
    
    return df

### 2. Search Engine 

def ntlk_analysis(info):
    '''
    This function takes as input a string and converts it into a list of words, 
    removing punctuations and the most common English words from the list and reducing each word 
    of the list to its stem or root format.
    '''
    final_words = []
    tokens = word_tokenize(info.lower())
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    for token in tokens:
        if token not in stop_words and token.encode().isalpha():
            stemming_token = ps.stem(token)
            final_words.append(stemming_token)
            
    return final_words

### 2.1.1 Create your index!

def open_vocabulary(filename):
    '''
    This function converts the contents of a file into a dictionary type object.
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
    This function converts the contents of a file into a dictionary type object.
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
            
### 2.1.2 Execute the query

def search_match(query, df, vocabulary, inverted_index):
    '''
    This function searches which documents in the dataframe contain all words in the query.
    It returns a list of indexes, the result of the intersection.
    '''
    query = ntlk_analysis(query)
    match = {vocabulary[term]: [] for term in query}

    for key,list_of_values in inverted_index.items():
        if key in match:
            for value in list_of_values:
                if value not in match[key]:
                    match[key].append(value)
                    
    final_values = list(match.values())
    intersection = set.intersection(*map(set,final_values))
    
    return intersection


def visualize_result(df, array):
    '''
    This function allows the visualization of some specific indexes and fields in the dataframe.
    '''
    result = []
    for index in array:
        series = df[['placeName', 'placeDesc','placeURL']].loc[index]
        result.append(series)
    
    return pd.DataFrame(result)


### 2.2. Conjunctive query & Ranking score

def heap_top_k(k, array, reverse):
    '''
    This functions converts an array type object into a heapify structure and returns the top-k 
    elements in the array. 
    '''
    heapq.heapify(array) 
    if reverse:
        top_k = (heapq.nlargest(k, array, lambda x:-x[1]))
    else:
        top_k = (heapq.nlargest(k, array, lambda x:x[1])) 
    return top_k

def cosine_similarity(v1,v2):
    '''
    '''
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


def tfidf_search_match(query, df, vocabulary, tfidf_inverted_index, idf, k):
    '''
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

        
    k_nearest = heap_top_k(k, k_nearest, False)
    
    for idx,score in k_nearest:
        series = df[['placeName', 'placeDesc','placeURL']].loc[idx]
        series['cosineSimilarity'] = score
        result.append(series)
    
    return pd.DataFrame(result)


### 3. Define a new score!

def jaccard(list1, list2):
    '''
    This function computes the Jaccard index.
    '''
    return len(set(list1).intersection(list2))/len(set(list1).union(list2))

def new_score_jaccard(query, df, vocabulary, inverted_index, k):
    '''
    This function computes a new score of Search Engine based on Jaccard index. 
    It returns a dataframe consisting of the documents in the dataframe 
    with the highest score.
    '''
    intersection = search_match(query, df, vocabulary, inverted_index)
    query = ntlk_analysis(query)
    rank = []

    for idx in intersection:
        if type(df.loc[idx]['placeTags']) == str:
            tags = ntlk_analysis(df.loc[idx]['placeTags'])
            jaccard_idx = jaccard(tags, query)
            rank.append((idx,jaccard_idx))
        else:
            rank.append((idx, -1))
            
    rank = heap_top_k(k, rank, False)
    result = []
    for idx,score in rank:
        series = df[['placeName', 'placeDesc','placeURL']].loc[idx]
        series['newScore'] = score
        result.append(series)
    
    return pd.DataFrame(result)

