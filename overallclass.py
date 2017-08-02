# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:15:02 2017

@author: zx_pe
"""
import tweepy
from unidecode import unidecode
import csv
import nbclass
import cleaner
import em_mixture_bern
from overallclass import *
from cleaner import *
from nbclass import *
from em_mixture_bern import *
from sentiment_analysis_lexicon import *
import nltk
from nltk.corpus import opinion_lexicon
from nltk.tokenize import treebank
from nltk.sentiment.util import _show_plot



class ExpediaTextAnalysis:
    def __init__(self):
        self.NB = NBClassifier()
        self.EM = em_mixture_bernoulli()
        self.lexicon_classifier = lexicon_classifier()
#        self.kmeans = KMeans()

    def loadUnlabelledData(self, filepath):
        dataset = []
        with open(filepath, encoding='ISO-8859-1') as f:
             dataset = [[line.rstrip("\n")] for line in f]
        self.dataset = dataset

    def loadLabelledData(self, filepath):
        dataset = []
        with open(filepath, encoding='ISO-8859-1') as trainfile:
             trainreader = csv.reader(trainfile)
             for row in trainreader:
                 dataset.append(row)
        self.labelledDataset = dataset
        
    def createTwitterAPI(self, cons_key,cons_secret,acc_token,acc_secret):
        auth = tweepy.OAuthHandler(cons_key, cons_secret)
        auth.set_access_token(acc_token, acc_secret)
        
        self.api = tweepy.API(auth)
        
    def scrapeTweets(self, username='ExpediaUK'):        
        expedia_cursor = tweepy.Cursor(self.api.user_timeline, id=username, tweet_mode="extended")
        expedia_tweets = []
        for page in expedia_cursor.pages(100):
            for tweet in page:
                expedia_tweets.append(unidecode(tweet.full_text))
        self.dataset = expedia_tweets
        with open("expedia_tweets.txt", "w") as output:
            output.write('\n'.join(expedia_tweets))
            
    def scrapeAtTweets(self, username='ExpediaUK'):
        at_expedia_cursor = tweepy.Cursor(self.api.search, q="to:{}".format(username), tweet_mode="extended") #Edit
        at_expedia_tweets = [unidecode(tweet.full_text) for tweet in at_expedia_cursor.items()]
        self.dataset = [[tweet] for tweet in at_expedia_tweets]
        with open("at_expedia_tweets.txt", "w") as output:
            output.write('\n'.join(at_expedia_tweets))
            
    def NBTrain(self):
        self.NB.loadData(self.labelledDataset)
        self.NB.trainTestSplit()
        self.NB.train()
        self.NB.predict()
        self.NB.frequencies()
    
    def NBOutput(self):
        self.NB.printAccuracy()
        self.NB.plotConfusionMatrix()
        
    def LC(self):
        self.lexicon_classifier.loadData(self.dataset)
        self.lexicon_classifier.predict()
    
    def EMtrain(self, k, min_count = 5, numit = 9999):
        self.EM.loadData(self.dataset, min_count)
        self.EM.train(k, max_it = numit)