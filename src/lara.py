#!/usr/bin/env python
# coding: utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import libs
# imports needed libs
import gensim 
import pandas as pd
import pickle
import itertools
import spacy
import time
import statistics

Aspects = ['atmosphere','food','location','service','staff','value']

# Util Functions
from util_funcs import *
from prepare_data import process_text_str
from sentiment_analysis import get_sentiment_score
                
def get_app_aspects():
    return read_pickle('pickles/aspect_review_app.pk')

def assign_aspect(sent,aspect_map):
    printTS(f"Called: assign_aspect {len(sent)}")
    
    dicty = {}
    
    stopwords = get_stop_word_list() + get_negation_word_list()
    try:
        stopwords.remove('restaurant')
        stopwords.remove('place')
    except:
        pass
        
    senti_model = read_pickle('pickles/aspect_senti_model.pk')
    nlp = spacy.load('en_core_web_sm', disable=["ner"])

    for word in nlp(sent.lower()):
        
        if word.lemma_ not in stopwords:
            score = 0
            aspec = None
            for asp in aspect_map.keys():
                try:
                    sco = senti_model.wv.similarity(w1=asp,w2=word.lemma_)
                    if sco > score:
                        score=sco
                        aspec=asp
                        dicty[aspec]= dicty.setdefault(asp, 0) + score    
                except:
                    continue
            #print(f"word {word} got {score} score")
            
    if len(dicty)==0:
        return None
    
    ret = max(dicty.items(), key=lambda k: k[1])
    if ret[1]<0.3:
        print(f"Info: Retuen value {ret} too low")
        return None
    
    return aspect_map[ret[0]]

def give_aspects_to_text(str_txt):
    printTS(f"Called: LARA give_aspects_to_text {len(str_txt)}")
    
    txt = str_txt
    
    aspect_map = get_aspect_map()
    topic_map  = get_all_topics_map()

    
    Aspect_Terms = Aspects
    
    vocab = []
    for x in topic_map.values():
        for y in x.split('/'):
            vocab.append(y)
    
    mydict = {}

    for w in Aspect_Terms:
        mydict[w] = mydict.setdefault(w, '')  
    
    for x in txt.replace('\\n','.').replace('\n','.').split('.'):
        if(len(x)>0):
            y = assign_aspect(x.replace('restaurant','location').replace('place','location'),aspect_map)
            mydict[y] = mydict.setdefault(y, '') + x +'.\n' 
    
    mydict.pop(None, None)
    
    for asp in Aspects:
        mydict[asp]=str(mydict[asp]).strip().replace('\n','').replace('..','.').replace(';','.').replace('.','.\n')
    
    mydict['name'] = 'N/A'
    mydict['review_id'] = 'N/A'
    mydict['text'] = str_txt
    mydict['review_stars'] = get_sentiment_score(process_text_str(str_txt))
    try:
        mydict['sentScore'] = int(mydict['review_stars'])
    except:
        mydict['sentScore'] = 3
    
    for asp in Aspects:
        mydict[asp+'Score'] = None
        mydict[asp+'Score'] = 0
        
        if str(mydict[asp]).replace('\n','') != '':
            try:
                mydict[asp+'Score'] = get_sentiment_score(str(mydict[asp]).replace('\n',''))
            except:
                mydict[asp+'Score'] = 3
    
    mydict = adjust_aspect_scores(mydict)
    mydict['review_stars'] = 'N/A'
    return (mydict)
    
    
def give_selected_aspects(df):
    printTS(f"Called: LARA give_selected_aspects {len(df)}")
    if len(df) != 1:
        printTS("None or Multiple App Aspects retreived")
        return {}
    
    df  = df.fillna('').astype(str)
    dfd = df.head(1).to_dict('records')[0]
    
    for asp in Aspects:
        dfd[asp]=str(dfd[asp]).strip().replace('\n','').replace('..','.').replace(';','.').replace('.','.\n')
    
    try:
        dfd['sentScore'] = get_sentiment_score(dfd['text'])
    except:
        dfd['sentScore'] = 3
        
    #dfd['sentScore'].astype(int)
    for asp in Aspects:
        dfd[asp+'Score'] = None
        dfd[asp+'Score'] = 0
        
        if str(dfd[asp]).replace('\n','') != '':
            try:
                dfd[asp+'Score'] = get_sentiment_score(str(dfd[asp]).replace('\n',''))
            except:
                dfd[asp+'Score'] = 3

    return adjust_aspect_scores(dfd)
    
def adjust_aspect_scores(dicty):
    printTS(f"Called: LARA adjust_aspect_scores {len(dicty)}")
    actScore = int(dicty['review_stars'])
    prdScore = int(dicty['sentScore'])
    fodScore = int(dicty['foodScore'])
    ambScore = int(dicty['atmosphereScore'])
    locScore = int(dicty['locationScore'])
    srvScore = int(dicty['serviceScore'])
    stfScore = int(dicty['staffScore'])
    valScore = int(dicty['valueScore'])
    
    totScore = 30
    newScore = 0
    
    try:
        newScore = int(round((prdScore+actScore)/2.0))
    except Exception as e:
        printTS(f"LARA Exception: {e}")
    
    try:
        totScore = int(round(statistics.mean(filter(None, [fodScore,ambScore,srvScore,stfScore,locScore,valScore]))))
    except Exception as e:
        dicty['text'] = dicty['text']+'\n\n<u>[<font color="red">Review text too short or not in english</font>]</u>'
        printTS(f"LARA Exception: {e}")
    
    dicty['sentScore']       = 0
    dicty['foodScore']       = 0
    dicty['atmosphereScore'] = 0
    dicty['locationScore']   = 0
    dicty['serviceScore']    = 0
    dicty['staffScore']      = 0
    dicty['valueScore']      = 0
    
    try:
        dicty['sentScore']       = int(round(prdScore))
        dicty['foodScore']       = int(round(newScore* fodScore/totScore))
        dicty['atmosphereScore'] = int(round(newScore* ambScore/totScore))
        dicty['locationScore']   = int(round(newScore* locScore/totScore))
        dicty['serviceScore']    = int(round(newScore* srvScore/totScore))
        dicty['staffScore']      = int(round(newScore* stfScore/totScore))
        dicty['valueScore']      = int(round(newScore* valScore/totScore))
        
                
        
    except Exception as e:
        printTS(f"LARA Exception: {e}")
    
    for asp in Aspects:
        if dicty[asp+'Score']>5:
            dicty[asp+'Score']=5
        if dicty[asp+'Score']<1 and dicty[asp] != '':
            dicty[asp+'Score']=1
    
    for asp in Aspects:
        if dicty[asp+'Score']==0:
            dicty[asp+'Score']=''
        else:
            dicty[asp+'Score']=str(dicty[asp+'Score'])
    return dicty

