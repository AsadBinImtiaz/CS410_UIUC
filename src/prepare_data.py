#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install pymongo


# In[2]:


#pip install gensim


# In[3]:


#pip install pandas


# In[4]:


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

# In[5]:


# import data from MongoDB
DBClient = MongoClient()
yelp_data = DBClient.yelp


# In[6]:


# Select business having atleast 50 reviews
min_review_count = 50

# businesses to Analyse
businesses_to_analyse = 'Restaurants'

# S3 Bucket
access_key_id="AKIAS6LZOC5VADNJTXS7"
secret_key_id="aNV7W7oWviWop7+HZKr6RCSUVJ7QCyw6wSYxhI9L"
bucket_arn_id="cs410-yelp/"
bucket_region="N. Virginia"

bucket =  's3://'+bucket_arn_id

os.environ['AWS_ACCESS_KEY_ID'] = access_key_id
os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key_id
os.environ['AWS_DEFAULT_REGION']='us-east-1'


# In[7]:


get_ipython().run_cell_magic('time', '', '# Get all restaurant businesses\nRestaurant_business = pd.DataFrame(yelp_data.business.find({"categories":{"$regex" :".*"+businesses_to_analyse+".*"}, "review_count":{"$gte":min_review_count} },  {\'business_id\':1, \'name\':1, \'city\':1, \'state\':1, \'stars\':1, \'review_count\':1, \'categories\':1, \'_id\': 0}))')


# In[8]:


get_ipython().run_cell_magic('time', '', "# Get all reviews\nAll_reviews = pd.DataFrame(yelp_data.review.find({},{'review_id':1, 'user_id':1, 'business_id':1, 'stars':1, 'useful':1, 'text':1, 'date':1, '_id': 0}))")


# In[9]:


get_ipython().run_cell_magic('time', '', "# Find all restaurant reviews\n#Restaurant_reviews = All_reviews[All_reviews.business_id.isin(Restaurant_business.business_id.values)]\nRestaurant_reviews = pd.merge(Restaurant_business,All_reviews, on='business_id').rename(columns={'stars_x':'business_stars', 'stars_y':'review_stars'})")


# In[12]:


get_ipython().run_cell_magic('time', '', '# Sample 5 Restaurant\nRestaurant_business.head(5)')


# In[10]:


get_ipython().run_cell_magic('time', '', "# Convert text to Unicode\nRestaurant_reviews['text'] = Restaurant_reviews['text'].map(lambda x: x.encode('unicode-escape','strict').decode('utf-8').replace('\\\\u',''))\n\n#Restaurant_reviews['text'] = Restaurant_reviews['text'].map(lambda x: re.sub(r'[^\\x00-\\x7f]',r'', x.encode('unicode-escape','strict').decode('utf-8')).replace('\\\\u',''))\n#Restaurant_reviews['text'] = Restaurant_reviews[u'text'].map(lambda x: re.sub(r'[^\\x00-\\x7f]',r'', x.encode('ascii', 'ignore').decode('utf-8')))\n\n#  Stats\n## Wall time: 8.3 s")


# In[11]:


get_ipython().run_cell_magic('time', '', "# Convert name to Unicode\nRestaurant_reviews['name']  = Restaurant_reviews['name'].map(lambda x: x.encode('unicode-escape','strict').decode('utf-8').replace('\\\\u',''))\nRestaurant_business['name'] = Restaurant_business['name'].map(lambda x: x.encode('unicode-escape','strict').decode('utf-8').replace('\\\\u',''))\n\n\n#  Stats\n## Wall time: 2.63 s")


# In[15]:


# Sample 5 Reviews
Restaurant_reviews.head(5)


# In[16]:


# Sample 5 Restaurants
Restaurant_business.head(5)


# In[17]:


# Write selected Restaurants to file
Restaurant_reviews.to_csv('processed_data/restaurant_reviews.csv',encoding='utf-8',line_terminator='\r')


# In[18]:


# Write selected Restaurant-reviews to file
Restaurant_business.to_csv('processed_data/restaurants.csv',encoding='utf-8',line_terminator='\r')


# #### Some charts on the loaded data, just for fun

# In[19]:


# plot how many reviews we have of each star
star_x = Restaurant_reviews.review_stars.value_counts().index
star_y = Restaurant_reviews.review_stars.value_counts().values

plot.figure(figsize=(8,5))
# colors are in the order 5, 4, 3, 1, 2
bar_colors = ['darkgreen', 'mediumseagreen', 'gold', 'crimson', 'orange']
plot.bar(star_x, star_y, color=bar_colors, width=.6)
plot.xlabel('Stars (Rating)')
plot.ylabel('Number of Reviews')
plot.title(f'Number of Reviews Per Rating of {businesses_to_analyse}')


# #### Breakdown of restaurants per state

# In[20]:


Restaurant_business.groupby('state').count()


# In[21]:


restaurants_per_state = Restaurant_business.groupby('state').count()[['business_id']].rename(columns={'state': 'State', 'business_id': 'Restaurants'})


# In[22]:


restaurants_per_state.sort_values(by='Restaurants').plot.bar(figsize=(10,10))


# In[38]:


Restaurant_AZ = pd.DataFrame(yelp_data.business.find({"categories":{"$regex" :".*"+businesses_to_analyse+".*"}, "review_count":{"$gte":min_review_count}, "state":"AZ" },  {'business_id':1, 'name':1, 'city':1, 'state':1, 'stars':1, 'review_count':1, 'categories':1, '_id': 0}))


# In[39]:


Restaurant_AZ_reviews = pd.merge(Restaurant_AZ,All_reviews, on='business_id').rename(columns={'stars_x':'business_stars', 'stars_y':'review_stars'})
Restaurant_AZ_reviews['text'] = Restaurant_AZ_reviews[u'text']


# In[40]:


Restaurant_AZ_reviews['text']  = Restaurant_AZ_reviews['text'].map(lambda x: x.encode('unicode-escape','strict').decode('utf-8').replace('\\u',''))
Restaurant_AZ_reviews['name']  = Restaurant_AZ_reviews['name'].map(lambda x: x.encode('unicode-escape','strict').decode('utf-8').replace('\\u',''))
Restaurant_AZ['name'] = Restaurant_AZ['name'].map(lambda x: x.encode('unicode-escape','strict').decode('utf-8').replace('\\u',''))


# In[44]:


Restaurant_AZ_reviews.to_csv(bucket+'processed_data/restaurant_il_reviews.csv',encoding='utf-8',line_terminator='\r')


# In[45]:


Restaurant_AZ.to_csv(bucket+'processed_data/restaurants_il.csv',encoding='utf-8',line_terminator='\r')


# In[46]:


Restaurant_AZ_reviews.shape


# In[47]:


# plot how many reviews we have of each star
star_x = Restaurant_AZ_reviews.review_stars.value_counts().index
star_y = Restaurant_AZ_reviews.review_stars.value_counts().values

plot.figure(figsize=(8,5))
# colors are in the order 5, 4, 3, 1, 2
bar_colors = ['darkgreen', 'mediumseagreen', 'gold', 'crimson', 'orange']
plot.bar(star_x, star_y, color=bar_colors, width=.6)
plot.xlabel('Stars (Rating)')
plot.ylabel('Number of Reviews')
plot.title(f'Number of Reviews Per Rating of {businesses_to_analyse}')


# ## Tokenization and Parsing
# 
# Read preprocessed restaurant and Review files.
# For testing, we read only the restaurants and reviews in **Arizona**

# In[5]:


# for now we restrich Restaurants to this number to develop the code
sample_restaurants_to_load = 100000

# Only Arizona Businesses, Change if needed
#restaurant_file='processed_data/restaurants_az.csv'
#reviews_file   ='processed_data/restaurant_az_reviews.csv'
restaurant_file='processed_data/restaurants.csv'
reviews_file   ='processed_data/restaurant_reviews.csv'


# In[6]:


get_ipython().run_cell_magic('time', '', '# SPACY\n# This is the large Spacy English Library\nnlp  = spacy.load(\'en_core_web_lg\')\nnlp2 = spacy.load(\'en_core_web_lg\', disable=["ner"])')


# *All stopword in restaurant reviews*

# In[7]:


# Stopwords for topic mining
stopwords = [line.rstrip('\n') for line in open('config/stopwords.txt', 'r', encoding='utf-8')]


# In[8]:


negations = [line.rstrip('\n') for line in open('config/negations.txt', 'r', encoding='utf-8')]


# *All stopword in restaurant names*

# In[9]:


# The words that appear in names of the Restaurants
# Restaurants name may appear multiple time in review, increasing its word frequenty
# For topic mining per restaurant, it is not useful and should be removed
# However words such as 'chicken' when come in restaurant name should be retained
stopnames = [line.rstrip('\n').lower() for line in open('config/names.txt', 'r')]


# In[10]:


get_ipython().run_cell_magic('time', '', "# Read Businesses\nall_restaurants = pd.read_csv(restaurant_file).drop(labels='Unnamed: 0', axis=1).head(sample_restaurants_to_load)")


# In[11]:


get_ipython().run_cell_magic('time', '', "# Read all reviews\nall_reviews = pd.read_csv(reviews_file).drop(labels='Unnamed: 0', axis=1).drop(labels='city', axis=1).drop(labels='state', axis=1).drop(labels='categories', axis=1).drop(labels='user_id', axis=1).drop(labels='date', axis=1)")


# In[12]:


get_ipython().run_cell_magic('time', '', '# Retain reviews of selected Businesses\nall_reviews = all_reviews[all_reviews.business_id.isin(all_restaurants.business_id)]')


# In[13]:


get_ipython().run_cell_magic('time', '', '# Top 5 Reviews\nall_reviews.head()')


# ##### Tokenization

# In[14]:


def tokenize_docs(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True re


# ##### Remove new line and spaces

# In[15]:


# String List cleaning, removes spaces, new lines
def clean_string(data):
    start = time.time()
    data = [str(sent).lower().replace('\\n',' ').replace('\n',' ').replace('.',' . ').replace(',',' , ').replace('?',' . ').replace('!',' . ') for sent in data]
    data = [str(sent).replace(';',' . ').replace('\r',' ').replace(':',' . ').replace('/',' / ').replace('"','').replace('$',' dollars ') for sent in data]
    data = [str(sent).replace('~','').replace('(','').replace(')','').replace('+','').replace('#','').replace('-','_').replace('%',' dollars ') for sent in data]
    data = [str(sent).strip('*').strip('-').replace('=',' ').replace('@',' ').replace('^',' ') for sent in data]
    print(f"lower cased       - took {time.time() - start:>{9.6}} secs")
    start = time.time()
    data = [re.sub('\s+', ' '  ,  str(sent)) for sent in data]
    print(f"spaces removed    - took {time.time() - start:>{9.6}} secs")
    start = time.time()
    data = [re.sub("n't", ' not', str(sent)) for sent in data]
    print(f"nots corrected    - took {time.time() - start:>{9.6}} secs")
    start = time.time()
    data = [sent.split() for sent in data]
    #data = list(tokenize_docs(data))
    print(f"tokenized         - took {time.time() - start:>{9.6}} secs")
    start = time.time()
    data = [[tok for tok in sent if tok not in stopwords ] for sent in data]
    print(f"Stopwords removed - took {time.time() - start:>{9.6}} secs")
    return data


# ##### Remove stopwords from restautant names
# we need to remove restaurant names from reviews, otherwise these may potentially become topics (most frequent *nouns*). But restaurant names can have other words, such as chinese, grill etc. which should not be removed from reviews
# In below function, we cleanse restautant name so that only valid parts should be removed. This consistes of proper nouns whaich are not in stopwords for reataurant names.

# In[16]:


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


# In[17]:


all_restaurants


# In[18]:


data = (all_reviews['text'])


# In[19]:


data[0]


# In[20]:


get_ipython().run_cell_magic('time', '', 'data = clean_string(data)\n\n#  Stats\n## lower cased       - took    48.314 secs\n## spaces removed    - took   157.966 secs\n## nots corrected    - took   13.8229 secs\n## tokenized         - took   513.574 secs\n## Stopwords removed - took   6836.05 secs\n## Wall time: 2h 6min 9s')


# In[21]:


## Checkpoint
#with open('data_list', mode="wb") as outfile:  # also, tried mode="rb"
#    outfile.write('\n'.join([','.join([tok for tok in rev]) for rev in data]))


# In[22]:


get_ipython().run_cell_magic('time', '', 'bigram  = gensim.models.Phrases(data, min_count=4, threshold=50) # higher threshold fewer phrases.\ntrigram = gensim.models.Phrases(bigram[data],min_count=3, threshold=100)  \n\n#  Stats\n## Wall time: 56min 4s')


# In[23]:


get_ipython().run_cell_magic('time', '', 'bigram_mod  = gensim.models.phrases.Phraser(bigram)\ntrigram_mod = gensim.models.phrases.Phraser(trigram)\n\n#  Stats\n## Wall time: 16min 49s')


# In[24]:


get_ipython().run_cell_magic('time', '', 'bigrams  = [bigram_mod[doc] for doc in data]\ntrigrams = [trigram_mod[bigram_mod[doc]] for doc in data]\n\n#  Stats\n## Wall time: 1h 27min 43s')


# In[25]:


#%%time
#with open ("processed_data/vocab.csv","w",encoding='utf-8')as vocab:
#    vocab.write('\n'.join(list(sorted(set(reduce(operator.concat, trigrams))))))
#    
#  Stats
## Several Hours


# In[26]:


get_ipython().run_cell_magic('time', '', 'all_reviews[\'topic_text\'] = [" ".join(trigram).replace(" .",".\\n") for trigram in trigrams]\n\n#  Stats\n## Wall time: 37.4 s')


# In[27]:


## Checkpoint
all_reviews.to_csv('processed_data/trigram_reviews.csv',encoding='utf-8')
#all_reviews = pd.read_csv('processed_data/cleaned_reviews.csv').drop(labels='Unnamed: 0', axis=1)


# ##### Remove stopwords restautant reviews
# In below function, we cleanse restautant reviews for **Topic Modelling**. We revove all stop words, keep only nouns, verbs, adjectives and advesbs, and remove restautant references in reviews.

# In[28]:


def clean_doc(doc,name_toks,allowed_postags=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV']):
    
    # Remove punctuation, symbols (#) and stopwords
    topic_allowed_postags=['PROPN', 'NOUN', 'VERB']
    sent_allowed_postags=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV']
    
    #toks = [tok.lemma_ for tok in doc ]
    #doc = nlp2(" ".join(toks))
    
    #for noun_phrase in doc.noun_chunks:
    #    if len(str(noun_phrase).split())>1 and len(str(noun_phrase).split())<3 and '_' not in str(noun_phrase):
    #        for x in str(noun_phrase).split():
    #            if x in stopwords or '_' in x:
    #                break;
    #            print(noun_phrase)
    #[noun_phrase.merge(noun_phrase.root.tag_, noun_phrase.root.lemma_, noun_phrase.root.ent_type_) for noun_phrase in doc.noun_chunks if len(str(noun_phrase).split())>1 and len(str(noun_phrase).split())<4 and '_' not in str(noun_phrase)]
    
    #doc = [tok.text for tok in doc if (tok.text.lower() not in stopwords and tok.pos_ != "PUNCT" and tok.pos_ != "SYM")]
    
    sents  = []
    
    for sent in doc.sents:
        sent_words = []
        for i,token in enumerate(sent):
            lemma = token.lemma_.strip().replace('_',' ')
            word  = token.text.replace('_',' ')
            pos   = token.pos_
            if word not in stopwords and word not in name_toks and pos != "PUNCT":
                sent_words.append(lemma.replace(' ','_'))
        if len(sent_words)>0:
            sent_words.append('.')
        sents.append(" ".join(sent_words))
    
    new_text = nlp2(" ".join(sents))
    
    topics = []
    sentis = []
    
    skip = False
    for sent in new_text.sents:
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
                    else:
                        sent_words.append(lemma.replace(" ","_"))
        if len(sent_words)>0:
            sent_words.append('.')
        sentis.append(" ".join(sent_words).replace(" .","."))
            
    topic_text = " ".join(topics)
    
    sentiment_text = " ".join((value for value in sentis if value != '.'))
    
    return [topic_text, sentiment_text]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'total = len(all_restaurants)\ncleansed_text = []\nstart = time.time()\nfor index, restaurant in all_restaurants.iterrows():\n    #print(f\'Cleaning reviews for restaurant: "{restaurant["name"]:<{40}}" [{index+1:>{5}}/{total:>{5}}]\')\n    if index % 500 == 0:\n        end = time.time()\n        print(f\'Cleaning reviews [{index+1:>{5}}/{total:>{5}} ] - {str(end-start):>{9.6}} secs\')\n        start = time.time()\n    \n    # Convert to list\n    data = all_reviews.query(\' business_id == "\'+restaurant[\'business_id\']+\'" \')[\'topic_text\']\n    \n    # iterate list, clean sentences\n    for parsed_review in nlp.pipe(iter(data), batch_size=1000, n_threads=8):\n        #[noun_phrase.merge(noun_phrase.root.tag_, noun_phrase.root.lemma_, noun_phrase.root.ent_type_) for noun_phrase in parsed_review.noun_chunks if len(str(noun_phrase).split())>1 and len(str(noun_phrase).split())<4]\n        cleansed_text.append(clean_doc(parsed_review,clean_name(restaurant["name"])))\n        #pprint(parsed_review)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "all_reviews['topic_text'] = [el[0] for el in cleansed_text]\nall_reviews['sentiment_text'] = [el[1] for el in cleansed_text]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "all_reviews.to_csv('processed_data/cleaned_reviews.csv',encoding='utf-8')\nall_restaurants.to_csv('processed_data/cleaned_restaurants.csv',encoding='utf-8')")


# In[ ]:


sorted(set(reduce(operator.concat, all_reviews['topic_text']).split()))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'with open ("processed_data/vocab.csv","w",encoding=\'utf-8\')as vocab:\n    vocab.write(\'\\n\'.join(list(sorted(set(reduce(operator.concat, all_reviews[\'topic_text\']).split())))))')


# In[217]:


#END

