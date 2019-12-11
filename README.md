# Yummy opinion advisor - Yelp restaurant topics and sentiment analysis

## 1. Introduction
The `Yummy opinion advisor (Yummy NLP)` is the anme of our Project. In this project, we analyse Yelp reviews of **Restaurants** and perform various NLP taks sucsh as **Topic Mining**, **Sentiment Analysis** and **Latent Aspect Rating Analysis**.

This project is contributed by:

- Asad Bin Imtiaz (aimtiaz2@illinois.edu)
- Karun Veluru (kveluru2@illinois.edu)
- Ron Swan (rdswan2@illinois.edu)

This project consists of 2 parts:

1> Processing the data and building up models for NLP tasks
2> Using the models built in first task, perform adhoc NLP in a web APP

The first task is hereafter referred as **Analytics** part of this project, while the later is named as **WebAPP** part.

### 1.1. Lequired Libraries

To perform NLP on Yelp reviews following python libraries are used:

- Spacy
- Pandas
- MongoDB
- Numpy
- Gensim
- SKLearn
- NLTK
- Seaborn
- S3FS
- pickle
- RE
- Time
- Functools
- String
- Math

Also be sure to download the sapcy english language libraries:
- en_core_web_lg
- en_core_web_sm

### 1.2 Setup & Installation (Analytics part only)

To install or setup the project (Analytics part, see definition above), you need to download the Yelp Open dataset from here: https://www.yelp.com/dataset/download
This data set contains the reviews of various kings of businesses and consists of several JSON file, which are around **8.5GB** in total. 

The data set zipped archive (tar-ball) is approximately 3.7GB, which uncompresses into around 8GB of Yelp data including Yelp business descriptions, their reviews, the users and tips. The data is composed of 6 json data files. These files should be extracted in the dataset directory. The JSON data files are ignored from Git versining by having an ignore entry in gitignore file.

The dataset needs to be downloaded and imported into a Mongo DB Instance. The data can also be downloaded unzipped from the projects S3 Bucket here: https://cs410-yelp.s3.amazonaws.com/dataset/. The downloaded data must be saved as **.json** files in the dataset directory. Once it is there, it can be loaded by executing `dataimport.bat` or `dataimport.sh.

### 1.3 Data Gathering
First the Yelp reviews need to be downloaded. There can be downloaded as a zippped achive from: https://www.yelp.com/dataset/download
The data set zipped archive (tar-ball) as around 3.7GB, which uncompresses into around 8GB of Yelp data including Yelp business descriptions, their reviews, the users and tips.
The data is composed of 6 jason data files. These files should be extracted in the dataset directory. The JSON data files are ignored from Git versining by having an ignore entry in `gitignore` file.

Once the data is stored in dataset directory, it is needed to be read into an instance of mongo DB. Make sure the mongo DB is installed and running during the data preperation phase. Mongo DB can be downloaded (for windows and other OS) from: https://www.mongodb.com/download-center/community?jmp=docs. The x64 Windows msi file is around 250MB large. Install the software with default settings. For convinience, the service can be executed with network user. After installation, add the MongoDB executables in the system PATH variavle. On windows the installations is ususally found at `C:\Program Files\MongoDB\Server\4.2\bin` directory. Once the MongoDB server is installed, is running and PATH variable is adjusted, the data can be loaded by executing `dataimport.bat` or `dataimport.sh` (depending on OS). Running this script will add all data in an instance named `yelp` (photos data will not be loaded).

## 2. Data Pre-Processing

The `prepare_data.ipynb` will read the data from mongo DB and combine reviews and businesses data into one csv file in processed_data directory (you may need to install pymongo). The businesses of catory 'Restaurants' will be considered only and all restaurants will be selected which have atleast 50 rewies. The data will be preprocessed with pandas into `processed_data/restaurant_reviews.csv` file (~2.5GB).

The preprocessing step takes the raw review text data, removes the stopword and URLs from it, parse and lemmatize all the sentences, combine frequent words appearing together as phrases (bi-grams/tri-grams) and keeps only relavant POS tags which are necassary for an NLP task, such as it Keeps Nouns, Propernouns and Verbs in the text for topic mining, but keeps Adjectives, adverbs and determination in addition for the text for sentiment analysis.

The `prepare_data.ipynb` will read the data from mongo DB and combine reviews and businesses data into one csv file in processed_data directory (you may need to install pymongo). The businesses of category 'Restaurants' will be considered only and all restaurants will be selected which have atleast 50 reviews. From MongoDB, we select relevant data and write to csv files to be used for next steps

The selection criteria is:

Restaurant businesses
Businesses with 50 or more reviews
Select only following fields from businesses:
-   'business_id'
-   'name'
-   'city'
-   'state'
-   'stars'
-   'review_count'
-   'categories'

We select only following fields from reviews:
-   'review_id'
-   'user_id'
-   'business_id'
-   'stars'
-   'useful'
-   'text'
-   'date'

The data from Restaurants and Reviews is merged in a single file, called `restaurant_reviews.csv`. All qualifying restaurant data is written to `restaurants.csv` file. There files are stored in `procesed_data` diectory.

### 2.1 Tokenization & Parsing

We read preprocessed restaurant and Review files. We remove all restaurant names from reviews. If we do the topic modelling on reviews for a restaurant, the name of the restaurant may appear as frequet topic. This is not ideal, so we will remove it from the review. But some words such as chicken can be in the name and should be retained in the review text. We remove stop words. These stopwords are manually prepared by us for this project. It includes some extra words such as 'restaurant', which we want to avoid for out Topic Modelling and Sentiment analysis. It also excludes words such as 'not' to do a better sentiment analysis. These include words such as not, niether, rarely etc. which change the sense of sentiment and should be considered in phrases are not considered are stop words in our case. All URLs are also removed. 

All frequent groups of words are converetd into BiGrams and TriGrams. The processed data is also stored as CVS files in `procesed_data` directory

## 3. Topic Mining

The `topic_mining.ipynb` we do the topic mining. We consider reviews with score 4 & 5 as positive, reviews with scores 1 & 2 as negative and the reviews with score 3 as neutral. We create three ***Non-Negative Matrix Factorization NMF*** models, one for each positive, one for negative and one for all of the reviews. 

The trained model is dumped in `pickels` directory for later use in webapp.

## 4. Sentiment Analysis

In `sentiment_analysis.ipynb` notebook, we to use the pre-processed text find out the sentiment. We build a **MultiLayer Perceptron** model with 90/10 train test split of cleansed sentiment relavant review text with 15000 words/phrases as features. The model accuracy for positive and negative reviews was 90% while for neutral reviews was 75%.

The trained model is dumped in `pickels` directory for later use in webapp.

## 5. LARA 

In `aspect_word_vector.ipynb` notebook, we will segment out review text to segments and assign an aspect with each segments. We build a word2vector model to to find similarity between differen words. All 6 aspects are splitted in 100 sub aspects for accuracy.

## 6. WebApp

The web App can be lauvhed by running `app.py` script. Once up and running, the app can be opened in browser using http://localhost:5000/ url.

## 7. Project structure

**src** folder contains all webapp source code
**template** folder contains all html flask templates
**dataset** folder contains raw dataset downloaded
**processed_data** folder contains all processed CVS files
**pickels** contains all pickeld classifiers and models
**config** folder contains all configuration and mapping files
