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

## Project structure
huh !!!

## Some heading
blah blah

### Some Sub heading
more blah blah blah

#### More Headings
even more blah blah blah
