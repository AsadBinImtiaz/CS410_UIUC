#!/usr/bin/env python
# coding: utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Import necessary libs
import pandas as pd
import re
import numpy as np
import time
from pprint import pprint
from functools import reduce 
import operator
import os

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Spacy
import spacy

# Plotting
import matplotlib.pyplot as plot

# S3 Access
import s3fs

# Mongo DB
from pymongo import MongoClient

# Util Functions
from util_funcs import *

# ## Data pre-processing
# 
# We have loaded the data in *Mongo DB* (with pandas we can also read jsons directly, but it needs **10x** more time)
# From from the mongo DB, we select relevant data and write in csv files to be used for next steps
# 
# The selection goes like this:
# * On restaurant businesses
# * Only Busenesses with 50 or more reviews 
# * Select only following fields from businesses:
#     - 'business_id'
#     - 'name'
#     - 'city'
#     - 'state'
#     - 'stars'
#     - 'review_count'
#     - 'categories'
# * Select only following fields from reviews:
#     - 'review_id'
#     - 'user_id'
#     - 'business_id'
#     - 'stars'
#     - 'useful'
#     - 'text'
#     - 'date'
#     
# In the following, we connect to MongoDb and load data in Pandas Data Frames. The data from Restaurants and Reviews is merged in a single file, called **restaurant_reviews.csv**. All qualifying restaurant data is written in **restaurants.csv** file.

dir_start   = 'src/'
dir_data    = 'processed_data/'
dir_config  = 'config/'
dir_pickles = 'pickles'

# S3 Bucket
access_key_id="AKIAS6LZOC5VADNJTXS7"
secret_key_id="aNV7W7oWviWop7+HZKr6RCSUVJ7QCyw6wSYxhI9L"
bucket_arn_id="cs410-yelp/"
bucket_region="N. Virginia"

bucket =  's3://'+bucket_arn_id

os.environ['AWS_ACCESS_KEY_ID']     = access_key_id
os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key_id
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

####################################################
# LOAD SPACY
# This is the large Spacy English Library
nlp  = spacy.load('en_core_web_lg')
nlp2 = spacy.load('en_core_web_lg', disable=["ner"])
####################################################

# Only Arizona Businesses, Change if needed
#restaurant_file='processed_data/restaurants_az.csv'
#reviews_file   ='processed_data/restaurant_az_reviews.csv'
restaurant_file='../processed_data/restaurants.csv'
reviews_file   ='../processed_data/restaurant_reviews.csv'


# ## Tokenization and Parsing
# 
# Read preprocessed restaurant and Review files.
# For testing, we read only the restaurants and reviews in **Arizona**

# for now we restrich Restaurants to this number to develop the code
sample_restaurants_to_load = 100000

# ##### Remove stopwords from restautant names
# we need to remove restaurant names from reviews, otherwise these may potentially become topics (most frequent *nouns*). But restaurant names can have other words, such as chinese, grill etc. which should not be removed from reviews
# In below function, we cleanse restautant name so that only valid parts should be removed. This consistes of proper nouns whaich are not in stopwords for reataurant names.

# *All stopword in restaurant reviews*
# Stopwords for topic mining
stopwords = get_stop_word_list()

negations = get_negation_word_list()

# *All stopword in restaurant names*
# The words that appear in names of the Restaurants
# Restaurants name may appear multiple time in review, increasing its word frequenty
# For topic mining per restaurant, it is not useful and should be removed
# However words such as 'chicken' when come in restaurant name should be retained
stopnames = get_stop_name_list()


def clean_name(name):
    name_toks = []
    
    # Nlp doc from Name
    name_doc = nlp2(name)
    for token in name_doc:
        
        # Retain Proper nouns in Name
        if token.pos_ == 'PROPN' or token.like_num:
        
            # Lose stop words in Name
            if token.text.lower() not in stopnames:
            
                # All Restaurant name tokens to be remoed from reviews of this reataurant
                name_toks.append(token.text.lower())
    
    #for noun_phrase in list(name_doc.noun_chunks):
        #if(len(str(noun_phrase).split())<2):
            #noun_phrase.merge(noun_phrase.root.tag_, noun_phrase.root.lemma_, noun_phrase.root.ent_type_)
    
    
    for chunk in name_doc.ents:
        name_toks.append(chunk.text.lower())
    
    return name_toks

# String List cleaning, removes spaces, new lines
def clean_string(data):
    data = remove_urls(data)
    data = remove_newlines(data)
    data = remove_spaces(data)
    data = remove_short_nots(data)
    data = split_on_space(data)
    data = remove_stop_words(data)
    return data

# Read Businesses
def load_restaurants_from_file(restaurant_file):
    return pd.read_csv(restaurant_file).drop(labels='Unnamed: 0', axis=1).head(sample_restaurants_to_load)
    
# Read all reviews
def load_reviews_from_file(restaurant_file):
    return pd.read_csv(restaurant_file).drop(labels='Unnamed: 0', axis=1).head(sample_restaurants_to_load)

def filter_review_for_restaurants(df_restaurants,df_reviews):
    return df_reviews[df_reviews.business_id.isin(df_restaurants.business_id)]

def pre_process_data(df_reviews):
        
    data = clean_string(df_reviews['text'])

    return data
    
def make_bigrams_trigrams(data):
    
    start = time.time()
    bigram  = gensim.models.Phrases(data, min_count=4, threshold=50) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data],min_count=3, threshold=100)  
    #  Stats
    ## Wall time: 56min 4s'


    bigram_mod  = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    #  Stats
    ## Wall time: 16min 49s


    bigrams  = [bigram_mod[doc] for doc in data]
    trigrams = [trigram_mod[bigram_mod[doc]] for doc in data]
    #  Stats
    ## Wall time: 1h 27min 43s
    printTS(f"Bi/TriGrams made  - took {time.time() - start:>{7.6}} secs")
    
    return [" ".join(trigram).replace(" .",".\n") for trigram in trigrams]

#all_reviews['topic_text'] = [" ".join(trigram).replace(" .",".\n") for trigram in trigrams]
#  Stats
## Wall time: 37.4 s
#

# ##### Remove stopwords restautant reviews
# In below function, we cleanse restautant reviews for **Topic Modelling**. We revove all stop words, keep only nouns, verbs, adjectives and advesbs, and remove restautant references in reviews.

def split_doc(doc):
    
    # Remove punctuation, symbols (#) and stopwords
    topic_allowed_postags=['PROPN', 'NOUN', 'VERB']
    sent_allowed_postags=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV', 'DET', 'ADP']
    
    topics = []
    sentis = []
    
    skip = False
    for sent in doc.sents:
        sent_words = []
        for i,token in enumerate(sent):
            if skip:
                skip = False
            else:
                lemma = token.lemma_.strip().replace('_',' ')
                word  = token.text.replace('_',' ')
                pos   = token.pos_
                if pos in topic_allowed_postags:
                    topics.append(lemma.replace(" ","_"))
                if pos in sent_allowed_postags:
                    if i+1<len(sent) and pos in ['ADJ', 'ADV'] and sent[i+1].pos_ in ['NOUN', 'VERB']:
                        sent_words.append(lemma+"_"+sent[i+1].lemma_)
                        skip = True
                    elif i+1<len(sent) and lemma in negations and sent[i+1].pos_ in ['ADJ', 'ADV'] and sent[i+1].lemma_ not in negations:
                        sent_words.append(lemma+"_"+sent[i+1].lemma_)
                        skip = True
                    elif len(lemma.replace(".",""))>1:
                        sent_words.append(lemma.replace(" ","_"))
        if len(sent_words)>0:
            sent_words.append('.')
        sentis.append(" ".join(sent_words).replace(" .","."))
            
    topic_text = str(" ".join(topics).replace("\n"," ")).replace("_","-")
    
    sentiment_text = str(" ".join((value for value in sentis if value != '.'))).replace("\n"," ").replace("_","-")
    
    return [topic_text, sentiment_text]


def clean_doc(doc, name_toks):
    
    sents  = []
    
    for sent in doc.sents:
        sent_words = []
        for i,token in enumerate(sent) :
            if token.lemma_ in list_diff([token.lemma_.lower() for token in sent],stopwords+name_toks) and token.lemma_ != "PUNCT":
                sent_words.append(str(token.lemma_))
        if len(sent_words)>0:
            sent_words.append('.')
        sents.append(" ".join(sent_words))
    
    new_doc = str(" ".join(sents).replace("  "," ").replace(" .","").replace(" .",".").replace(" .",".").replace("..","."))
    
    return split_doc(nlp2(new_doc))
    

def clean_text(all_restaurants, all_reviews):
    
    total = len(all_restaurants)
    
    printTS(f'Cleaning {str(total)} reviews')
    
    cleansed_text = []
    start = time.time()
    for index, restaurant in all_restaurants.iterrows():
        if index % 100 == 0:
            end = time.time()
            printTS(f'Cleaning reviews [{index+1:>{5}}/{total:>{5}} ] - {str(end-start):>{9.6}} secs')
            start = time.time()
        
        # Convert to list        
        data = all_reviews.query(' business_id == "'+restaurant['business_id']+'" ')['topic_text']
        data = [u''+str(txt) for txt in data]
        
        # iterate list, clean sentences
        for parsed_review in nlp2.pipe(iter(data), batch_size=5000, n_threads=20):
            #[noun_phrase.merge(noun_phrase.root.tag_, noun_phrase.root.lemma_, noun_phrase.root.ent_type_) for noun_phrase in parsed_review.noun_chunks if len(str(noun_phrase).split())>1 and len(str(noun_phrase).split())<4]
            cleansed_text.append(clean_doc(parsed_review,clean_name(restaurant["name"])))
    
    all_reviews['topic_text']     = [el[0] for el in cleansed_text]
    all_reviews['sentiment_text'] = [el[1] for el in cleansed_text]
    
    return cleansed_text

def process_text(df_restaurants, df_reviews):
    
    data = pre_process_data(df_reviews)
    
    data = make_bigrams_trigrams(data)
    
    df_reviews['topic_text'] = data
    
    return clean_text(df_restaurants,df_reviews)
    
def init_pre_processing(state_filter='IL', filePath='../processed_data/', business_file_name='restaurants.csv', review_file_name='restaurant_reviews.csv'):

    if state_filter != '':
        state_filter+='_'
        
    restaurant_file=filePath+state_filter+business_file_name
    reviews_file   =filePath+state_filter+review_file_name

    # Read Businesses
    all_restaurants = load_restaurants_from_file(restaurant_file)

    # Read all reviews
    all_reviews = load_reviews_from_file(reviews_file)
    
    # Retain reviews of selected Businesses
    all_reviews = filter_review_for_restaurants(all_restaurants,all_reviews)    
    
    printTS(f"Processing {len(all_restaurants)} reataurants ({len(all_reviews)} reviews)")
    cleaned_reviews = process_text(all_restaurants,all_reviews)
    
    all_reviews.to_csv(filePath+state_filter+'cleaned_reviews.csv',encoding='utf-8')
    all_restaurants.to_csv(filePath+state_filter+'cleaned_restaurants.csv',encoding='utf-8')

def load_data_from_mongo_db():
    mongo_data_to_csv()    
    
def process_text_str(review_text,restaurant_name=''):
    
    data = clean_string([review_text])
    data = make_bigrams_trigrams(data)
    
    total = len(data)
    
    cleansed_text = []
    start = time.time()

    data = [u''+str(txt) for txt in data]
        
    # iterate list, clean sentences
    for parsed_review in nlp2.pipe(iter(data), batch_size=5000, n_threads=20):
        #[noun_phrase.merge(noun_phrase.root.tag_, noun_phrase.root.lemma_, noun_phrase.root.ent_type_) for noun_phrase in parsed_review.noun_chunks if len(str(noun_phrase).split())>1 and len(str(noun_phrase).split())<4]
        cleansed_text.append(clean_doc(parsed_review,[restaurant_name]))
    
    printTS(f'Cleaning review   - took {str(time.time()-start):>{9.6}} secs')
    
    return cleansed_text[0]
    
if __name__ == "__main__":
    
    try:
        
        cleansed_text = process_text_str('Good food experience. I was with my family. Food was amazing. Service was slow.')
        
        printTS(f'*** Topic Text    : {cleansed_text[0]}')
        printTS(f'*** Sentiment Text: {cleansed_text[1]}')
        
    except Exception as SomeError:
        printTS (f"Pre Processing Falied: {str(SomeError)}")
        
        