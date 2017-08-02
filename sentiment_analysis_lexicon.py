#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 06:47:42 2017

@author: Miram
"""
import nltk
from nltk.corpus import opinion_lexicon
from nltk.tokenize import treebank
from nltk.sentiment.util import _show_plot
from pandas import DataFrame
from nltk.sentiment.util import mark_negation
# Sentiment Classification using Liu and Hu opinion lexicon.

## Negation Demo
#sentence = "I didn't like this hotel . It was bad ."
#if nltk.sentiment.vader.negated(sentence) == True:
#    sentence = ' '.join(mark_negation(sentence.split()))
#sentiment = demo_liu_hu_lexicon_updated(sentence, plot=True)


class lexicon_classifier:    
    def demo_liu_hu_lexicon_updated(self, sentence, plot=False):
        """
        Basic example of sentiment classification using Liu and Hu opinion lexicon.
        This function simply counts the number of positive, negative and neutral words
        in the sentence and classifies it depending on which polarity is more represented.
        Words that do not appear in the lexicon are considered as neutral.
    
        :param sentence: a sentence whose polarity has to be classified.
        :param plot: if True, plot a visual representation of the sentence polarity.
        """
        
    
        tokenizer = treebank.TreebankWordTokenizer()
        pos_words = 0
        neg_words = 0
        sentiment = ''
        tokenized_sent = [word.lower() for word in tokenizer.tokenize(sentence)]
    
        x = list(range(len(tokenized_sent))) # x axis for the plot
        y = []
    
        for word in tokenized_sent:
            if "_neg" in word:
                word = word.split("_", 1)[0]
                if word in opinion_lexicon.positive():
                    neg_words += 1
                    y.append(-1) # negative
                elif word in opinion_lexicon.negative():
                    pos_words += 1
                    y.append(+1) # positive
                else:
                    y.append(0) # neutral
            else:
                if word in opinion_lexicon.positive():
                    pos_words += 1
                    y.append(1) # positive
                elif word in opinion_lexicon.negative():
                    neg_words += 1
                    y.append(-1) # negative
                else:
                    y.append(0) # neutral
    
        if pos_words > neg_words:
            #print('Positive')
            sentiment = 'Positive'
        elif pos_words < neg_words:
            #print('Negative')
            sentiment = 'Negative'
        elif pos_words == neg_words:
            #print('Neutral')
            sentiment = 'Neutral'
    
        if plot == True:
            _show_plot(x, y, x_labels=tokenized_sent, y_labels=['Negative', 'Neutral', 'Positive'])
        
        return sentiment
        
    def loadData(self, x):
        self.dataset = x

    # Sentiment Classification handling negation
    def predict(self):
        predicted = []
        r = 0
        for row in self.dataset:
            r += 1
            print('Row {} of {}'.format(r, len(self.dataset)))
            sentence = str(row).strip('[]')
            if nltk.sentiment.vader.negated(sentence) == True:
                sentence = ' '.join(mark_negation(sentence.split()))
            prediction = self.demo_liu_hu_lexicon_updated(sentence, plot=False)
            predicted.append(prediction)
        self.predicted = predicted
    
        # Output predictions to csv file
        self.lexicon_predictions = DataFrame([row for row in self.dataset], self.predicted)
        print(self.lexicon_predictions.head(n=5))
        self.lexicon_predictions.to_csv("lexicon_predictions.csv")
        