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
import pickle as pickle
import warnings

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
access_key_id="AKIAS6LZOC5VOE3DTJCU"
secret_key_id="HMyfJFojkSvalPv3EGWC9tElUWivH1AhQFvWB0Vy"
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
dir_home   = ''

# Print message with Timestamp
def printTS(strInput):
    print (str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ': ' + str(strInput))
    logOutput(strInput)
    
def logOutput(strInput,newLineRep=""):
    #if logEnabled == 1:
    logging.info(str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ': ' + str(strInput).replace("\n",newLineRep))

log_file  = dir_home+"logs/"+str(datetime.now().strftime("%Y%m%d%H%M%S"))+".log"

def read_pickle(filename_with_path):
    pk = None
    if not os.path.isfile(dir_home+filename_with_path):
        printTS(f'File: {dir_home+filename_with_path} does not exists')
        download_file(filename_with_path)
    try:
        with open(dir_home+filename_with_path, 'rb') as f:
            pk = pickle.load(f)
    except:
        pass
    return pk
#
# download file from S3
#             
def download_file(filename_with_path):
    if os.path.isfile(dir_home+filename_with_path) and os.path.getsize(dir_home+filename_with_path)>0:
        printTS(f'File: {(dir_home+filename_with_path)} already locally exists')
        return
    else:
        printTS(f'Downloading {filename_with_path} from S3 Bucket: {bucket} to {os.path.abspath(dir_home+filename_with_path)}')
        with open(dir_home+filename_with_path,'wb') as fin:
            s3 = s3fs.S3FileSystem(anon=False)
            if s3.exists(bucket+filename_with_path):
                with s3.open(bucket+filename_with_path,'rb') as fout:
                    fin.write(fout.read())
                    printTS(f'File {filename_with_path} downloaded from S3 Bucket: {bucket}')
            else:
                printTS(f'File {filename_with_path} not exists on S3 Bucket: {bucket+filename_with_path} ')

#
# upload file to S3
#             
def upload_file(filename_with_path):
    if not os.path.isfile(dir_home+filename_with_path):
        printTS(f'File: {os.path.abspath(dir_home+filename_with_path)} does not locally exists')
        return
    else:
        printTS(f'Uploading {filename_with_path} to S3 Bucket: {bucket}')
        with open(dir_home+filename_with_path,'rb') as fout:
            s3 = s3fs.S3FileSystem(anon=False)
            if not s3.exists(bucket+filename_with_path):
                with s3.open(bucket+filename_with_path,'wb') as fin:
                    fin.write(fout.read())
                    printTS(f'File {filename_with_path} uploaded to S3 Bucket: {bucket}')
            else:
                printTS(f'File {filename_with_path} already exists on S3 Bucket: {bucket} path {filename_with_path}')

#
# upload file to S3
#             
def upload_file_anyway(filename_with_path):
    if not os.path.isfile(filename_with_path):
        printTS(f'File: {filename_with_path} does not locally exists')
        return
    else:
        printTS(f'Uploading {filename_with_path} to S3 Bucket: {bucket}')
        with open(dir_home+filename_with_path,'rb') as fout:
            s3 = s3fs.S3FileSystem(anon=False)
            with s3.open(bucket+filename_with_path,'wb') as fin:
                fin.write(fout.read())
                printTS(f'File {filename_with_path} uploaded to S3 Bucket: {bucket}')

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
    try:
        return [line.rstrip('\n').lower() for line in open(dir_home+'config/stopwords.txt', 'r', encoding='utf-8')]
    except:
        download_file('config/stopwords.txt')
        return [line.rstrip('\n').lower() for line in open(dir_home+'config/stopwords.txt', 'r', encoding='utf-8')]
        
#
# Get Negation Words
#
def get_negation_word_list():
    try:
        return [line.rstrip('\n').lower() for line in open(dir_home+'config/negations.txt', 'r', encoding='utf-8')]
    except:
        download_file('config/negations.txt')
        return [line.rstrip('\n').lower() for line in open(dir_home+'config/negations.txt', 'r', encoding='utf-8')]
    
#
# Get Negation Words
#
def get_stop_name_list():
    try:
        return [line.rstrip('\n').lower() for line in open(dir_home+'config/names.txt', 'r', encoding='utf-8')]
    except:
        download_file('config/names.txt')
        return [line.rstrip('\n').lower() for line in open(dir_home+'config/names.txt', 'r', encoding='utf-8')]

def get_get_config_map(filename_with_path):
    ret = None
    try:
        with open(dir_home+filename_with_path, 'r') as f:
            ret = {line.split(';')[0]: line.split(';')[1].replace('\n','') for line in f.readlines()}
    except:
        download_file(filename_with_path)
        with open(dir_home+filename_with_path, 'r') as f:
            ret = {line.split(';')[0]: line.split(';')[1].replace('\n','') for line in f.readlines()}
        
    return ret
    
# 
##### Tokenization
# 
def tokenize_docs(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True re

#
# Get Topic Map
#
def get_all_topics_map():
    ret = {}
    with open(dir_home+'config/all_topics.txt', 'r') as f:
         ret = {line.split(';')[0]: line.split(';')[1].replace('\n','') for line in f.readlines()}
    return ret

#
# Get Aspect Map
#
def get_aspect_map():
    return get_get_config_map('config/aspect_map.txt')
    #with open(dir_home+'config/aspect_map.txt', 'r') as f:
    #    ret = {line.split(';')[0]: line.split(';')[1].replace('\n','') for line in f.readlines()}
    #return ret
    
#
# Remove new lies and symbols & lowercase from list of data
#
def remove_newlines(data):
    start = time.time()
    data = [str(sent).lower().replace('\\n',' ').replace('\n',' ').replace('.',' . ').replace(',',' , ').replace('?',' . ').replace('!',' . ') for sent in data]
    data = [str(sent).replace(';',' . ').replace('\r',' ').replace(':',' . ').replace('/',' / ').replace('"','').replace('$',' dollars ') for sent in data]
    data = [str(sent).replace('~','').replace('(','').replace(')','').replace('+','').replace('#','').replace('-','_').replace('%',' dollars ') for sent in data]
    data = [str(sent).strip('*').strip('-').replace('=',' ').replace('@',' ').replace('^',' ') for sent in data]
    printTS(f"lower cased       - took {time.time() - start:9.6f} secs")
    return data

#
# Remove URLs from list of data
#    
def remove_urls (data):
    start = time.time()
    data = [re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', str(sent), flags=re.MULTILINE) for sent in data]
    printTS(f"URLs   removed    - took {time.time() - start:9.6f} secs")
    return(data)
    
#
# Remove spaces and symbols from list of data
#
def remove_spaces (data):
    start = time.time()
    data = [re.sub('\s+', ' '  ,  str(sent)) for sent in data]
    printTS(f"spaces removed    - took {time.time() - start:9.6f} secs")
    return data

#
# Convert n't to not in list of data
#
def remove_short_nots (data):
    start = time.time()
    data = [re.sub("n't", ' not', str(sent)) for sent in data]
    printTS(f"nots corrected    - took {time.time() - start:9.6f} secs")
    return data
    
#
# tokenize within list of data
#
def split_on_space (data):
    start = time.time()
    data = [sent.split() for sent in data]
    #data = list(tokenize_docs(data))
    printTS(f"tokenized         - took {time.time() - start:9.6f} secs")
    return data
    
#
# Remove stop words from list of data
#
def remove_stop_words(data):    
    start = time.time()
    stops = get_stop_word_list()
    data = [list_diff(sent,stops) for sent in data]
    printTS(f"Stopwords removed - took {time.time() - start:9.6f} secs")
    return data

def list_diff(list1,list2):
    return list(itertools.filterfalse(set(list2).__contains__, list1)) 
    

def tokenizer(x):
    return ( w for w in str(x).split() if len(w) >3)

def remove_stop_words_fast(data):
    start = time.time()
    stops = get_stop_word_list()
    printTS(f"Stopwords removed - took {time.time() - start:9.6f} secs")
    return [list_diff(sent,stops) for sent in data]

def mongo_data_to_csv(save_on_s3=False):
    
    printTS('Loading mongo data to CSV files')
    try:
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

def start_logger():
    warnings.filterwarnings("ignore")
    try:
        Path(log_file).touch()   
        logging.basicConfig(filename=log_file,level=logging.INFO)
        logEnabled = 1
        printTS ("------------------START--------------------")            
    except Exception as SomeError:
        printTS (f"Warning: logger initialization falied: {str(SomeError)}")
    
    
if __name__ == "__main__":
    start_logger()
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

