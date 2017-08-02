# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 15:18:59 2017

@author: Miram
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.sentiment.util import *
from sklearn.metrics import confusion_matrix
import plot_confusion_matrix
from plot_confusion_matrix import *
from nltk import pos_tag
import pandas as pd
from cleaner import *
import cleaner

class NBClassifier:
    def unique_list(self, l):
        ulist = []
        [ulist.append(x) for x in l if x not in ulist]
        return ulist
     
    # For each document we remove all duplicate words
    def remDuplicate(self, stemReviews):
        for row in stemReviews:
            sentence = row[1][1]
            words = sentence.split()
            row[1][1] = ' '.join(self.unique_list(words))
        return stemReviews
    # Loading the training dataset from labelled csv file
    
    def loadData(self, x):
        dataset = x                 
        df = DataFrame(dataset, columns=dataset[0])
        df1 = df[['stars','user_review']]
        df2 = df1.ix[1:]
        #df2.groupby('stars').count()
        
        # Collapse star ratings 2 & 3 into 1 and 4 into 5
        pd.options.mode.chained_assignment = None
        df2.loc[df2['stars'] == '2', 'stars'] = '1'
        df2.loc[df2['stars'] == '3', 'stars'] = '1'
        df2.loc[df2['stars'] == '4', 'stars'] = '5'
        #df2.groupby('stars').count()
        
        # Preprocess reviews
        for index, row in df2.iterrows():
            row['user_review'] = ' '.join(stem(tokenize(row['user_review'])))
        
        stemReviews = [review for review in df2.iterrows()]
        self.stemReviews = self.remDuplicate(stemReviews)
    # Function to remove duplicates from given list of words

    
    # Use stratified sampling to split the data into training and test sets
    def trainTestSplit(self):    
        X = [row[1][1] for row in self.stemReviews]
        y = [row[1][0] for row in self.stemReviews]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y)
    
    def train(self):
        # Define list of common terms (stop words)
        stopwords = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                     'to', 'was', 'were', 'will', 'with']
        
        # Feature extraction
        # Translate the input documents into vectors of features
        
        # Parameters:
        # 1- stop_words = list, exclude stop words from the resulting tokens
        # 2- min_df=5, discard words appearing in less than 5 documents
        # 3- max_df=0.8, discard words appering in more than 80% of the documents
        self.vectorizer = CountVectorizer(stop_words = stopwords,
                                     min_df = 5,
                                     max_df = 0.8)
        
        # Create the vocabulary (i.e. the list of words/features) 
        # and the feature weights from the training data
        train_vectors = self.vectorizer.fit_transform(self.X_train)
        
        #print(DataFrame(train_vectors.A, columns=vectorizer.get_feature_names()).to_string())
        self.results = DataFrame(train_vectors.A, columns=self.vectorizer.get_feature_names()) ####################
        #results.to_csv("/Users/Miram/Desktop/sentiment analysis/train_results.csv")
        
        # Use a naïve Bayes classifier to learn from the features
        self.classifier = MultinomialNB()
        
        # Train it by calling fit, passing in the feature vector and the target vector
        self.classifier.fit(train_vectors, self.y_train)
        
    def predict(self):
    # Try the classifier on the test data
        test_vectors = self.vectorizer.transform(self.X_test)
        # Our predictions vector should be [1, 0]
        # corresponding to sentiment classification (positive and negative)
        self.predictions = self.classifier.predict(test_vectors)
        self.test_predictions = DataFrame({'review': self.X_test, 'predicted': self.predictions, 'actual': self.y_test})
        print(self.test_predictions.head(n=5))
        #test_predictions.to_csv("/Users/Miram/Desktop/sentiment analysis/review_predictions.csv")
    
    def frequencies(self):
        # Find most common words for low review scores
        low_ratings = self.test_predictions.loc[self.test_predictions['predicted'] == '1', ]
        low_ratings = [row[1][2] for row in low_ratings.iterrows()]
        
        low_ratings_new = []
        for row in low_ratings:
            tokens = nltk.word_tokenize(row)
            tags = pos_tag(tokens)
            resultwords  = [t[0] for t in tags if t[1] != 'PRP' and t[1] != 'PRP$' 
                            and t[1] != 'CC' and t[1] != 'RB' and t[1] != 'WRB'
                            and t[1] != 'IN' and t[1] != 'DT' and t[1] != 'WDT']
            low_ratings_new.append(' '.join(resultwords))
        
        low_vectors = self.vectorizer.fit_transform(low_ratings_new)
        low_frequencies = [(word, low_vectors.getcol(index).sum()) for word, index in self.vectorizer.vocabulary_.items()]      
        self.sorted_low_frequencies = sorted (low_frequencies, key = lambda x: -x[1])
    #    print (sorted_low_frequencies)
        
        d1 = {}
        for x, y in self.sorted_low_frequencies:
            d1.setdefault(x, y)
        #print(d1)
        
        df1 = pd.DataFrame([d1])
        df1 = df1.T
        df1.columns = ['frequency']
        df1 = df1.sort(['frequency'], ascending=[False])
        df1.to_csv("low_word_frequencies.csv")
        
        # Find most common words for high review scores
        high_ratings = self.test_predictions.loc[self.test_predictions['predicted'] == '5', ]
        high_ratings = [row[1][2] for row in high_ratings.iterrows()]
        
        high_ratings_new = []
        for row in high_ratings:
            tokens = nltk.word_tokenize(row)
            tags = pos_tag(tokens)
            resultwords  = [t[0] for t in tags if t[1] != 'PRP' and t[1] != 'PRP$' 
                            and t[1] != 'CC' and t[1] != 'RB' and t[1] != 'WRB'
                            and t[1] != 'IN' and t[1] != 'DT' and t[1] != 'WDT']
            high_ratings_new.append(' '.join(resultwords))
            
        high_vectors = self.vectorizer.fit_transform(high_ratings_new)
        high_frequencies = [(word, high_vectors.getcol(index).sum()) for word, index in self.vectorizer.vocabulary_.items()]
        self.sorted_high_frequencies = sorted (high_frequencies, key = lambda x: -x[1])
    #    print (sorted_high_frequencies)
        
        d2 = {}
        for x, y in self.sorted_high_frequencies:
            d2.setdefault(x, y)
        #print(d2)
        
        df2 = pd.DataFrame([d2])
        df2 = df2.T
        df2.columns = ['frequency']
        df2 = df2.sort(['frequency'], ascending=[False])
        df2.to_csv("high_word_frequencies.csv")
    
    def printAccuracy(self):
        print("Accuracy: {}".format(accuracy_score(self.y_test, self.predictions)))
    
        # Compute confusion matrix
        self.cnf_matrix = confusion_matrix(self.y_test, self.predictions)
        np.set_printoptions(precision=2)
        print(self.cnf_matrix)
    
    
    def plotConfusionMatrix(self):
    #    Plot non-normalized confusion matrix
    #    plt.figure()
    #    plot_confusion_matrix(cnf_matrix, classes=["1", "5"],
    #                          title='Confusion matrix, without normalization')
        
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(self.cnf_matrix, classes=["1", "5"], normalize=True,
                              title='Normalized confusion matrix')
        
        plt.show()