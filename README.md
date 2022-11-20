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

In this repository can be found:

### 1. `stuff`

####
A folder containing some useful files that  must be loaded in order to run the code.

* For the data scrapping was used **final_dataset.csv**, a dataset we built with the required information in order to deal with more pratical operations;
* For the search engine were used **vocabulary.txt**, **inverted_index.txt** and **tfidf_inverted_index.txt**, the pre-processed documents during search engines;
* For the command line was used  **final_dataset.tsv** and was produced **CommandLine_result.png**, a screenshot of the results obtained in output;
* For the heoretical question was produced **TQ_final_output.txt**, a text file which allows a complete and better visualization of the result.

### 2. `main.ipynb`

#### 
A Jupyter notebook that contains all our answers and scripts relative to the HW's questions.

### 2. `scraper.py`

#### 
In this module can be found all the methods involved with the actual implementation of the scraping.

### 3. `engine.py`

#### 
In this module can be found all the methods involved with the actual implementation of the search engines: all the defined metrics, the actual search engine functions and some tools to help loading the data.

### 4. `command_line.sh`

#### 
A bash shell script file that contains the code ready to run in order to answer the command line question.

## Additional link for work visualization
https://nbviewer.org/github/LucaFabri99/3HW-ADM-Fabri.Dinino.Aur/blob/main/main.ipynb

**IMPORTANT:**
Use it to visualize the map that was produced in exercise 4.
