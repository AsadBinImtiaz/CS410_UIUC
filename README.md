# Yummy opinion advisor - Yelp restaurant topics and sentiment analysis

## Introduction
The `Yummy opinion advisor` ...

## Data Preperation
First the Yelp reviews need to be downloaded. There can be downloaded as a zippped achive from: https://www.yelp.com/dataset/download
The data set zipped archive (tar-ball) as around 3.7GB, which uncompresses into around 8GB of Yelp data including Yelp business descriptions, their reviews, the users and tips.
The data is composed of 6 jason data files. These files should be extracted in the dataset directory. The JSON data files are ignored from Git versining by having an ignore entry in `gitignore` file.

Once the data is stored in dataset directory, it is needed to be read into an instance of mongo DB. Make sure the mongo DB is installed and running during the data preperation phase. Mongo DB can be downloaded (for windows and other OS) from: https://www.mongodb.com/download-center/community?jmp=docs. The x64 Windows msi file is around 250MB large. Install the software with default settings. For convinience, the service can be executed with network user. After installation, add the MongoDB executables in the system PATH variavle. On windows the installations is ususally found at `C:\Program Files\MongoDB\Server\4.2\bin` directory. Once the MongoDB server is installed, is running and PATH variable is adjusted, the data can be loaded by executing `dataimport.bat` or `dataimport.sh` (depending on OS). Running this script will add all data in an instance named `yelp` (photos data will not be loaded).

The `prepare_data.ipynb` will read the data from mongo DB and combine reviews and businesses data into one csv file in processed_data directory (you may need to install pymongo). The businesses of catory 'Restaurants' will be considered only and all restaurants will be selected which have atleast 50 rewies. The data will be preprocessed with pandas into `processed_data/restaurant_reviews.csv` file (~2.5GB).
 

## Project structure
huh !!!

## Some heading
blah blah

### Some Sub heading
more blah blah blah

#### More Headings
even more blah blah blah