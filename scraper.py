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


###_________________________________HTML SCRAPER__________________________________


def string_to_datetime(string):
    '''
     This function converts an object of type string to one of type datetime.
    '''
    return str(datetime.strptime(string, '%B %d, %Y').date())


def darkAtlasScraper(text, filename):
    '''
    This function extracts places' information from the html documents of interest
    and builds a dictionary of these information for every place we go through.
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
    
    # Names
    try:
        scraped['placeName'] = soup.find_all('h1',{'class':'DDPage__header-title'})[0].contents[0]
    except IndexError:
        pass

    # Tags       
    try:
        scraped['placeTags'] = list(map(lambda s:s.strip(),
                                        [tag.contents[0] for tag in soup.find_all('a',{'class':'itemTags__link js-item-tags-link'})]))
    except IndexError:
        pass
    
    # Number of people who have been there 
    counters = soup.find_all('div',{'class':'title-md item-action-count'})
    try:
        scraped['numPeopleVisited'] = int(counters[0].contents[0])
    except IndexError:
        pass
    # Number of people who want to visit the place
    try:
        scraped['numPeopleWant'] = int(counters[1].contents[0])
    except IndexError:
        pass
    
    # Description      
    place_desc = ''
    for paragraph in soup.find_all('div',{'class':'DDP__body-copy'})[0].find_all('p'):
        for element in paragraph.contents:
            if re.search('<[^>]*>', str(element)):
                element = re.sub('<[^>]*>', "", str(element))
                place_desc += element
            else:
                place_desc += str(element)
    scraped['placeDesc'] = place_desc
    
    # Short description
    try:
        scraped['placeShortDesc'] = soup.find_all('h3',{'class':'DDPage__header-dek'})[0].contents[0].replace(u'\xa0', u'')
    except IndexError:
        pass

    # Nearby Places
    nearby = []
    try:
        for nearbies in soup.find_all('div',{'class':'DDPageSiderailRecirc__item-text'}):
            nearby.append(nearbies.find_all('div',{'class':'DDPageSiderailRecirc__item-title'})[0].contents[0])
        scraped['placeNearby'] = nearby
    except IndexError:
        pass
    
    # Addresses 
    try:
        address = (str(soup.find_all('aside',{'class':'DDPageSiderail__details'})[0]
                           .find_all('address',{'class':'DDPageSiderail__address'})[0]
                           .find_all('div')[0])
                           .split('\n', 1)[0])
        scraped['placeAddress'] = re.sub('<[^>]*>', " ", address)
    except IndexError:
        pass
    
    # Latitud and Longitude of the place's location
    coordinates = soup.find_all('div',{'class':'DDPageSiderail__coordinates js-copy-coordinates'})[0].contents[2]
    scraped['placeAlt'] = float(coordinates.split()[0][:-1])
    scraped['placeLong'] = float(coordinates.split()[1])


    # The username of the post editors
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

    # Post publishing date     
    try:
        scraped['placePubDate'] = string_to_datetime(soup.find_all('div',{'class':'DDPContributor__name'})[0].contents[0])
    except IndexError:
        pass

    # The names of the lists that the place was included in 
    # The names of the related places
    kircher = soup.find_all('div',{'class':'athanasius'})
    for piece in kircher:
        for piecer in piece.find_all('div',{'class':'CardRecircSection__title'}):
            if piecer.contents[0] == 'Related Places':
                scraped['placeRelatedPlaces'] = [re.sub('<[^>]*>', "", str(chunk.contents[1])) 
                                                 for chunk in piece.find_all('h3',{'class':'Card__heading --content-card-v2-title js-title-content'})]
            elif 'Appears in' in piecer.contents[0]:
                scraped['placeRelatedLists'] =  [re.sub('<[^>]*>', "", str(chunk.contents[1])) 
                                                 for chunk in piece.find_all('h3',{'class':'Card__heading --content-card-v2-title js-title-content'})]
    
    # The URL of the page of the place
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
