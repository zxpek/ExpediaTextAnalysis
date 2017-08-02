#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from: http://www.nltk.org/_modules/nltk/sentiment/util.html


"""

def demo_liu_hu_lexicon_updated(sentence, plot=False):
    """
    Basic example of sentiment classification using Liu and Hu opinion lexicon.
    This function simply counts the number of positive, negative and neutral words
    in the sentence and classifies it depending on which polarity is more represented.
    Words that do not appear in the lexicon are considered as neutral.

    :param sentence: a sentence whose polarity has to be classified.
    :param plot: if True, plot a visual representation of the sentence polarity.
    """
    from nltk.corpus import opinion_lexicon
    from nltk.tokenize import treebank
    from nltk.sentiment.util import _show_plot

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