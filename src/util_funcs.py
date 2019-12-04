#!/usr/bin/env python
# coding: utf-8

# Import necessary libs
import pandas as pd
import re
import numpy as np
import time
from pprint import pprint
from functools import reduce 
import operator
import os
import itertools
from datetime import datetime
from pathlib import Path

# Spacy
import spacy

# S3 Access
import s3fs

# Mongo DB
from pymongo import MongoClient

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# logging
import logging

#
# ## Helpful Functions
# 

# S3 Bucket
access_key_id="AKIAS6LZOC5VADNJTXS7"
secret_key_id="aNV7W7oWviWop7+HZKr6RCSUVJ7QCyw6wSYxhI9L"
bucket_arn_id="cs410-yelp/"
bucket_region="N. Virginia"

bucket =  's3://'+bucket_arn_id

os.environ['AWS_ACCESS_KEY_ID']     = access_key_id
os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key_id
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

dir_start   = 'src/'
dir_data    = 'processed_data/'
dir_config  = 'config/'
dir_pickles = 'pickles'

logEnabled = 0

# Print message with Timestamp
def printTS(strInput):
    print (str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ': ' + str(strInput))
    logOutput(strInput)
    
def logOutput(strInput,newLineRep=""):
    if logEnabled == 1:
        logging.info(str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ': ' + str(strInput).replace("\n",newLineRep))

log_file  = "../logs/"+str(datetime.now().strftime("%Y%m%d%H%M%S"))+".log"
             

#
# Save Dafaframe to CSV on S3 project bucket
#
def df_to_S3_csv(df,filename,dirpath='processed_data',encoding='utf-8',line_terminator='\r'):
    df.to_csv(bucket+dirpath+'/'+filename,encoding,line_terminator)

#
# Read Dafaframe from CSV on S3 project bucket
#
def S3_csv_to_df(filename,dirpath='processed_data'):
    return pd.read_csv(bucket+dirpath+'/'+filename).drop(labels='Unnamed: 0', axis=1)

#
# Get Stop Words
#
def get_stop_word_list():
    return [line.rstrip('\n').lower() for line in open('../config/stopwords.txt', 'r', encoding='utf-8')]
    
#
# Get Negation Words
#
def get_negation_word_list():
    return [line.rstrip('\n').lower() for line in open('../config/negations.txt', 'r', encoding='utf-8')]
    
#
# Get Negation Words
#
def get_stop_name_list():
    return [line.rstrip('\n').lower() for line in open('../config/names.txt', 'r', encoding='utf-8')]

# 
##### Tokenization
# 
def tokenize_docs(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True re


#
# Remove new lies and symbols & lowercase from list of data
#
def remove_newlines(data):
    start = time.time()
    data = [str(sent).lower().replace('\\n',' ').replace('\n',' ').replace('.',' . ').replace(',',' , ').replace('?',' . ').replace('!',' . ') for sent in data]
    data = [str(sent).replace(';',' . ').replace('\r',' ').replace(':',' . ').replace('/',' / ').replace('"','').replace('$',' dollars ') for sent in data]
    data = [str(sent).replace('~','').replace('(','').replace(')','').replace('+','').replace('#','').replace('-','_').replace('%',' dollars ') for sent in data]
    data = [str(sent).strip('*').strip('-').replace('=',' ').replace('@',' ').replace('^',' ') for sent in data]
    printTS(f"lower cased       - took {time.time() - start:>{9.6}} secs")
    return data

#
# Remove URLs from list of data
#    
def remove_urls (data):
    start = time.time()
    data = [re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', str(sent), flags=re.MULTILINE) for sent in data]
    printTS(f"URLs   removed    - took {time.time() - start:>{9.6}} secs")
    return(data)
    
#
# Remove spaces and symbols from list of data
#
def remove_spaces (data):
    start = time.time()
    data = [re.sub('\s+', ' '  ,  str(sent)) for sent in data]
    printTS(f"spaces removed    - took {time.time() - start:>{9.6}} secs")
    return data

#
# Convert n't to not in list of data
#
def remove_short_nots (data):
    start = time.time()
    data = [re.sub("n't", ' not', str(sent)) for sent in data]
    printTS(f"nots corrected    - took {time.time() - start:>{9.6}} secs")
    return data
    
#
# tokenize within list of data
#
def split_on_space (data):
    start = time.time()
    data = [sent.split() for sent in data]
    #data = list(tokenize_docs(data))
    printTS(f"tokenized         - took {time.time() - start:>{9.6}} secs")
    return data
    
#
# Remove stop words from list of data
#
def remove_stop_words(data):    
    start = time.time()
    stops = get_stop_word_list()
    data = [list_diff(sent,stops) for sent in data]
    printTS(f"Stopwords removed - took {time.time() - start:>{9.6}} secs")
    return data

def list_diff(list1,list2):
    return list(itertools.filterfalse(set(list2).__contains__, list1)) 
    

def remove_stop_words_fast(data):
    start = time.time()
    stops = get_stop_word_list()
    printTS(f"Stopwords removed - took {time.time() - start:>{9.6}} secs")
    return [list_diff(sent,stops) for sent in data]

def mongo_data_to_csv(save_on_s3=False):
    
    printTS('Loading mongo data to CSV files')
    try:
        dir_home    = '../'
        
        if save_on_s3:
            dir_home    = bucket
            printTS('File will be uploaded to S3 Bucket: '+bucket)    
        
        # import data from MongoDB
        DBClient = MongoClient()
        yelp_data = DBClient.yelp


        # Select business having atleast 50 reviews
        min_review_count = 50

        # businesses to Analyse
        businesses_to_analyse = 'Restaurants'

        #state_filter = 'IL'

        # Get all restaurant businesses\n
        Restaurant_business = pd.DataFrame(yelp_data.business.find({"categories":{"$regex" :".*"+businesses_to_analyse+".*"}, "review_count":{"$gte":min_review_count} }, {'business_id':1, 'name':1, 'city':1, 'state':1, 'stars':1, 'review_count':1, 'categories':1, '_id': 0}))

        # Get all reviews
        All_reviews = pd.DataFrame(yelp_data.review.find({},{'review_id':1, 'user_id':1, 'business_id':1, 'stars':1, 'useful':1, 'text':1, 'date':1, '_id': 0}))

        # Find all restaurant reviews
        #Restaurant_reviews = All_reviews[All_reviews.business_id.isin(Restaurant_business.business_id.values)]
        Restaurant_reviews = pd.merge(Restaurant_business,All_reviews, on='business_id').rename(columns={'stars_x':'business_stars', 'stars_y':'review_stars'})

        # Convert text to Unicode
        Restaurant_reviews['text']  = Restaurant_reviews['text'].map(lambda x: x.encode('unicode-escape','strict').decode('utf-8').replace('\\u',''))

        # Convert name to Unicode
        Restaurant_reviews['name']  = Restaurant_reviews['name'].map(lambda x: x.encode('unicode-escape','strict').decode('utf-8').replace('\\u',''))
        Restaurant_business['name'] = Restaurant_business['name'].map(lambda x: x.encode('unicode-escape','strict').decode('utf-8').replace('\\u',''))
        #  Stats
        ## Wall time: 2.63 s")
        
        printTS('Generating files')    #### Logging

        # Write selected Restaurants to file
        Restaurant_reviews.to_csv(dir_home+dir_data+'restaurant_reviews.csv',encoding='utf-8',line_terminator='\r')

        # Write selected Restaurant-reviews to file
        Restaurant_business.to_csv(dir_home+dir_data+'restaurants.csv',encoding='utf-8',line_terminator='\r')

        #Restaurant_AZ = pd.DataFrame(yelp_data.business.find({"categories":{"$regex" :".*"+businesses_to_analyse+".*"}, "review_count":{"$gte":min_review_count}, "state":state_filter },  {'business_id':1, 'name':1, 'city':1, 'state':1, 'stars':1, 'review_count':1, 'categories':1, '_id': 0}))

        #Restaurant_AZ_reviews = pd.merge(Restaurant_AZ,All_reviews, on='business_id').rename(columns={'stars_x':'business_stars', 'stars_y':'review_stars'})
        #Restaurant_AZ_reviews['text'] = Restaurant_AZ_reviews[u'text']
        
        #Restaurant_AZ_reviews['text']  = Restaurant_AZ_reviews['text'].map(lambda x: x.encode('unicode-escape','strict').decode('utf-8').replace('\\u',''))
        #Restaurant_AZ_reviews['name']  = Restaurant_AZ_reviews['name'].map(lambda x: x.encode('unicode-escape','strict').decode('utf-8').replace('\\u',''))
        #Restaurant_AZ['name'] = Restaurant_AZ['name'].map(lambda x: x.encode('unicode-escape','strict').decode('utf-8').replace('\\u',''))

        #Restaurant_AZ_reviews.to_csv(dir_home+dir_data+state_filter+'_restaurant_reviews.csv',encoding='utf-8',line_terminator='\r')

        #Restaurant_AZ.to_csv(dir_home+dir_data+state_filter+'restaurants.csv',encoding='utf-8',line_terminator='\r')
        
        printTS('Data loaded in CSV')
        
    except Exception as SomeError:
     printTS (f"Loading mongo data to CSV files falied: {str(SomeError)}")


    
if __name__ == "__main__":
    try:
        Path(log_file).touch()   
        logging.basicConfig(filename=log_file,level=logging.INFO)
        logEnabled = 1
        logOutput("------------------START--------------------")            
    except Exception as SomeError:
        printTS (f"Warning: logger initialization falied: {str(SomeError)}")
    
    printTS ("Initializated")

'''        
    data = S3_csv_to_df('cleaned_reviews_il.csv')['text']
    print('\n start \n\n')
    data[0] = data[0]+'\n Adding a URL at end (http://www.s.c/aa/dsa+aNV7W7oWviWop7 is so much fun). I shouldn\'t be so funny!\n\n'
    print (data[0])
    
    data = remove_urls(data)
    
    data = remove_newlines(data)
    
    data = remove_spaces(data)
    
    data = remove_short_nots(data)
    
    data = split_on_space(data)
    
    data = remove_stop_words_fast(data)
    
    data = [" ".join(sent) for sent in data]
    
    print ('\n\n'+data[0])
'''    
    
#END

