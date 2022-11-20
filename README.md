# 3HW-ADM-Fabri.Dinino.Aur

## Group Members
* Marina
* Leonardo 
* Luca 

## HW's goals
We were asked to create our own dataset by scraping the [Atlas Obsura](https://www.atlasobscura.com/) website, an American online magazine and travel firm, catalogues unusual and obscure tourist locations. The goals of the Homework are:

* be introduced to crawling and parsing of html documents;
* learn how to create a search engine that whould suggets locations based on the user's query and how to use metrics to evalute its output (such as cosine similarity, tfIdf score and Jaccard similarity);
* become familiar with the command line;
* make acquaintance of sorting algorithms.

## Files description
**NOTE:**
The first part of the homework has been runned locally in Leopardo's machine. It was about gathering the dataset through some scraping scripts, so following the guidelines we drained the links and then downloaded the HTML and parsed them, targeting some specific information. 
In this repository you'll find:

### 1. `stuff`

####
A folder in which they are loacted some useful files which must be loaded if you want to run the code:

* For the scapping part, we can find **vocabulary.txt**, **inverted_index.txt** and **tfidf_inverted_index.txt**, the pre-processed documents during search engines;
* For the search engining part, we can find **final_dataset.csv**, a dataset we build with the required information in order to deal with more practical operations;
* For the command line part, we can find **CommandLine_result.png**, a screenshot of the result obtained in output, and **final_dataset.tsv**, dataset in tsv format used for the command line.

### 2. `main.ipynb`

#### 
A Jupyter notebook that contains all our answers and scripts relative to the HW's questions. 

### 2. `scraper.py`

#### 
All the methods implied in this first part are in the **scraper.py** file.

### 3. `engine.py`

#### 
In this module we can find all the methods involved with the actual implementation of the search engines. So here there are all the metrics we defined, the actual search engine functions and some tools to help loading the data.

### 4. `command_line.sh`

#### 
A bash shell script file that contains the prepared script to answer to the command line question.

## Additional link for work visualization
https://nbviewer.org/github/LucaFabri99/3HW-ADM-Fabri.Dinino.Aur/blob/main/main.ipynb
