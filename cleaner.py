# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 00:14:57 2017

@author: zx_pe
"""

from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import string
import pandas as pd

def tokenize(tweet):
    tk = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokened = tk.tokenize(tweet.lower())
    listWords = [word for word in tokened if word not in set(string.punctuation)]
    return listWords

def stem(listWords):
    stemmer = PorterStemmer()
    stemWords = [stemmer.stem(word) for word in listWords]        
    return stemWords
    
def BuildStopWords(corpus, min_count = 5):
    stop = stopwords.words('english')
    counts = Counter(corpus)
    for word,count in counts.items():
        if count < min_count:
            stop.append(word)
    stop = set(stop)
    filteredCorpus = set([word for word in corpus if word not in stop])
    return (filteredCorpus, stop)

def filterCount(tweet, stop):
    filteredTweet = [w for w in tweet if w not in stop]
    pTweet = Counter(filteredTweet)
    return(pTweet)
    
def PreprocessData(tweetList, min_count = 5):
    stemTweet = [stem(tokenize(tweet[0])) for tweet in tweetList]    
    corpus = [item for sublist in stemTweet for item in sublist]
    filteredCorpus, stop = BuildStopWords(corpus, min_count)
    df = pd.DataFrame(0, index=range(len(stemTweet)), columns = filteredCorpus) 
    pTweets = [filterCount(tweet, stop) for tweet in stemTweet]
    for i in range(len(pTweets)):
        for word in pTweets[i]:
            df.iloc[i][word] = pTweets[i][word]
    return(df)
    
def PreprocessBinaryData(tweetList, min_count = 5):
    stemTweet = [stem(tokenize(tweet[0])) for tweet in tweetList]    
    corpus = [item for sublist in stemTweet for item in sublist]
    filteredCorpus, stop = BuildStopWords(corpus, min_count)
    df = pd.DataFrame(0, index=range(len(stemTweet)), columns = filteredCorpus) 
    pTweets = [filterCount(tweet, stop) for tweet in stemTweet]
    for i in range(len(pTweets)):
        for word in pTweets[i]:
            df.iloc[i][word] = 1
    return(df)