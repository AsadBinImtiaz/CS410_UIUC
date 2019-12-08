#!/usr/bin/env python
# coding: utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import libs
import pandas as pd
import seaborn as sns
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import pickle

# Util Functions
from util_funcs import *
from prepare_data import process_text_str

def get_vocab():
    return read_pickle('pickles/senti_vocab.pk')

def get_MLP_Classifier():
    return read_pickle('pickles/senti_mlp_classifier.pk')
    
def text_process(text):
    stop_words = get_stop_word_list()
    stopwords = frozenset(stop_words)
    
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords]

def get_sentiment_score(str_txt):
    vocab = get_vocab()
    mlp   = get_MLP_Classifier()
    
    pr_t = vocab.transform([str_txt])
    
    return mlp.predict(pr_t)[0]

if __name__ == '__main__':
    printTS('Analyzer started')
    txt = process_text_str("We've always been there on a Sunday so we were hoping that Saturday dim sum would be less busy. No such luck. We were surprised that some of the dishes were cold because it was so packed; I could understand if it was empty and the carts weren't circulating but every table was full. It took a while to get drinks and other items (napkins). The dishes were not of the same quality as they had been on other visits, but they were acceptable.")
    printTS(txt)
    printTS(get_sentiment_score('ok'))
    