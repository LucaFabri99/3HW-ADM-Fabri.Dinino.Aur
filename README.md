# 3HW-ADM-Fabri.Dinino.Aur

## Group Members
* Marina
* Leonardo 
* Luca 

## HW's goals
We were asked to create our own dataset by scraping the [Atlas Obscura](https://www.atlasobscura.com/) website, an American online magazine and travel firm, and to focus on the [Most popular places](https://www.atlasobscura.com/places?sort=likes_count) section.

The goals of the Homework are:

* leaning how to crawl and parse the html documents;
* learning how to create a search engine that suggests locations based on the user's query and how to use metrics to evalute its output (such as cosine similarity, tfIdf score and Jaccard similarity);
* becoming familiar with the command line;
* making acquaintance with the sorting algorithms

## Files description
**IMPORTANT:**
The first part of the homework has been runned locally in Leopardo's machine. It was about gathering the dataset through some scraping scripts, so following the guidelines we drained the links and then downloaded the HTML and parsed them, targeting some specific information. 

In this repository you'll find:

### 1. `stuff`

####
A folder in which are located some useful files; these must be loaded if you want to run the code.

* For the data scapping part, we use **final_dataset.csv**, a dataset we have built with the required information in order to deal with more practical operations;
* For the search engine part, we use: **vocabulary.txt**, **inverted_index.txt** and **tfidf_inverted_index.txt**, the pre-processed documents during search engines;
* For the command line part, we can find **CommandLine_result.png**, a screenshot of the results obtained in output, and **final_dataset.tsv**, dataset in tsv format used for the command line.

### 2. `main.ipynb`

#### 
A Jupyter notebook that contains all our answers and scripts relative to the HW's questions. 

### 2. `scraper.py`

#### 
In this module we can find all the methods involved with the actual implementation of the scraping.

### 3. `engine.py`

#### 
In this module we can find all the methods involved with the actual implementation of the search engines. So here there are all the metrics we defined, the actual search engine functions and some tools to help loading the data.

### 4. `command_line.sh`

#### 
A bash shell script file that contains code ready to answer the command line question.

## Additional link for work visualization
https://nbviewer.org/github/LucaFabri99/3HW-ADM-Fabri.Dinino.Aur/blob/main/main.ipynb
