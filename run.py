# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 22:36:34 2017

@author: zx_pek
"""
%run cleaner.py
%run em_mixture_bern.py
%run nbclass.py
%run twitterkeys.py
%run overallclass.py
%run sentiment_analysis_lexicon.py
ETA = ExpediaTextAnalysis()
ETA.loadUnlabelledData('Data\\expedia_tweets.txt')
ETA.loadLabelledData('Data\\reviews.csv')

ETA.NBTrain()
ETA.NBOutput()

ETA.LC()

ETA.EMtrain(5, min_count= 100)


