import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import math
import logging
import logging.config
import os
import yaml
import time
import datetime
import pickle
import boto3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
# %matplotlib inline
nltk.download('stopwords')
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
REVIEW_TEXT_COLUMN = 'cleansed_text'
az_reviews_df = 'dataaz_reviews_df.pkl'
vocab_file = 'datavocab.pkl'
transformed_x_file = 'datatransformed_x.pkl'

logger = logging.getLogger(__name__)


def setup_logging(default_path='logging-config.yml',
                  default_level=logging.INFO,
                  env_key='LOG_CFG')
    path = default_path
    value = os.getenv(env_key, None)
    if value
        path = value
    if os.path.exists(path)
        with open(path, 'rt') as f
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else
        logging.basicConfig(level=default_level)


def read_s3_bucket(bucket, data_key)
    data_location = 's3{}{}'.format(bucket, data_key)

    chunksize = 1000000
    chunk_list = []
    df_chunk = pd.read_csv(data_location, chunksize=chunksize)
    for chunk in df_chunk
        chunk_list.append(chunk)

    data = pd.concat(chunk_list)
    return data


def read_az_reviews()
    bucket = 'cs410-yelp'
    data_key = 'processed_datareviews_az.csv'

    df = read_s3_bucket(bucket, data_key)
    df = df.drop(labels='Unnamed 0', axis=1)
    df['review_stars'] = df['review_stars'].astype(int)
    df[REVIEW_TEXT_COLUMN] = df[REVIEW_TEXT_COLUMN].astype(str)
    with open(az_reviews_df, wb) as file
        pickle.dump(df, file)
    return df


def init_classification(df)
    # CLASSIFICATION
    df_classes = df[(df['review_stars'] == 1)  (df['review_stars'] == 3) 
                    (df['review_stars'] == 5)]
    logger.info(df_classes.head())
    logger.info(Shape of df {}.format(df_classes.shape))

    # Seperate the data set into X and Y for prediction
    x = df_classes[REVIEW_TEXT_COLUMN]
    y = df_classes['review_stars']
    logger.info(x.head())
    logger.info(y.head())
    return x, y


def text_process(text)
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [
        word for word in nopunc.split()
        if word.lower() not in stopwords.words('english')
    ]


def init_transformation(x)
    logger.info(init_transformationstart-time ---  {}.format(
        datetime.datetime.now().time()))
    start_time = time.time()

    r0 = x[1]
    logger.info(Transformation of x[0] - r[0], r0)
    logger.info(r0)

    vocab = CountVectorizer(analyzer=text_process).fit(x)
    logger.info(Length of Vocabulary {}.format(len(vocab.vocabulary_)))
    with open(vocab_file, wb) as file
        pickle.dump(vocab, file)
    vocab0 = vocab.transform([r0])
    logger.info(Vocab0 {}.format(vocab0))

    logger.info(init_transformationend-time ---  {}.format(
        datetime.datetime.now().time()))
    logger.info(Elapsed {} seconds.format(round(time.time() - start_time,
                                                   4)))
    return vocab


def vectorize(vocab, xx)
    logger.info(vectorizestart-time ---  {}.format(
        datetime.datetime.now().time()))
    start_time = time.time()

    x = vocab.transform(xx)
    with open(transformed_x_file, wb) as file
        pickle.dump(x, file)
    #Shape of the matrix
    logger.info(Shape of the sparse matrix {}.format(x.shape))
    #Non-zero occurences
    logger.info(Non-Zero occurences {}.format(x.nnz))

    # DENSITY OF THE MATRIX
    density = (x.nnz  (x.shape[0]  x.shape[1]))  100
    logger.info(Density of the matrix {}.format(density))

    logger.info(vectorizeend-time ---  {}.format(
        datetime.datetime.now().time()))
    logger.info(Elapsed {} seconds.format(round(time.time() - start_time,
                                                   4)))
    return x


def print_results(y_true, y_pred, classifier_name)
    logger.info(Confusion Matrix for {}.format(classifier_name))
    logger.info(confusion_matrix(y_true, y_pred))
    logger.info(Classification Report)
    logger.info(
        classification_report(y_true,
                              y_pred,
                              target_names=['1-Star', '3-Star', '5-Star']))
    logger.info(Score {}n.format(
        round(accuracy_score(y_true, y_pred)  100, 2)))


def multinomial_naive_bayes(x_train, x_test, y_train, y_test)
    logger.info(multinomial_naive_bayesstart-time ---  {}.format(
        datetime.datetime.now().time()))
    start_time = time.time()

    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)
    predmnb = mnb.predict(x_test)
    print_results(y_test, predmnb, Multinomial Naive Bayes)

    logger.info(multinomial_naive_bayesend-time ---  {}.format(
        datetime.datetime.now().time()))
    logger.info(Elapsed {} seconds.format(round(time.time() - start_time,
                                                   4)))
    return mnb


def random_forest_classifier(x_train, x_test, y_train, y_test)
    logger.info(random_forest_classifierstart-time ---  {}.format(
        datetime.datetime.now().time()))
    start_time = time.time()

    rmfr = RandomForestClassifier()
    rmfr.fit(x_train, y_train)
    p = rmfr.predict(x_test)
    print_results(y_test, p, Random Forest Classifier)

    logger.info(random_forest_classifierend-time ---  {}.format(
        datetime.datetime.now().time()))
    logger.info(Elapsed {} seconds.format(round(time.time() - start_time,
                                                   4)))
    return rmfr


def SVM(x_train, x_test, y_train, y_test)
    logger.info(SVMstart-time ---  {}.format(
        datetime.datetime.now().time()))
    start_time = time.time()

    svm = SVC(random_state=101)
    svm.fit(x_train, y_train)
    p = svm.predict(x_test)
    print_results(y_test, p, SVM)

    logger.info(SVMend-time ---  {}.format(
        datetime.datetime.now().time()))
    logger.info(Elapsed {} seconds.format(round(time.time() - start_time,
                                                   4)))
    return svm


def multilayer_perceptron(x_train, x_test, y_train, y_test)
    logger.info(multilayer_perceptronstart-time ---  {}.format(
        datetime.datetime.now().time()))
    start_time = time.time()

    mlp = MLPClassifier()
    mlp.fit(x_train, y_train)
    p = mlp.predict(x_test)
    print_results(y_test, p, Multilayer Perceptron)

    logger.info(multilayer_perceptronend-time ---  {}.format(
        datetime.datetime.now().time()))
    logger.info(Elapsed {} seconds.format(round(time.time() - start_time,
                                                   4)))
    return mlp


def kNN(x_train, x_test, y_train, y_test)
    logger.info(kNNstart-time ---  {}.format(
        datetime.datetime.now().time()))
    start_time = time.time()

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train, y_train)
    p = knn.predict(x_test)
    print_results(y_test, p, kNN)

    logger.info(kNNend-time ---  {}.format(
        datetime.datetime.now().time()))
    logger.info(Elapsed {} seconds.format(round(time.time() - start_time,
                                                   4)))
    return knn


def decision_tree(x_train, x_test, y_train, y_test)
    logger.info(decision_treestart-time ---  {}.format(
        datetime.datetime.now().time()))
    start_time = time.time()

    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    p = dt.predict(x_test)
    print_results(y_test, p, Decision Tree)

    logger.info(decision_treeend-time ---  {}.format(
        datetime.datetime.now().time()))
    logger.info(Elapsed {} seconds.format(round(time.time() - start_time,
                                                   4)))
    return dt


def test_rating(df, classifier, item, rating_type, vocab)
    logger.info(rating_type + start)
    item = 2
    pr = df[REVIEW_TEXT_COLUMN][item]
    logger.info(pr)
    logger.info(nActual Rating {}.format(df['review_stars'][item]))
    pr_t = vocab.transform([pr])
    logger.info(Predicted Rating {}.format(classifier.predict(pr_t)[0]))
    logger.info(rating_type + end)


def upload_files_to_s3()
    S3 = boto3.client('s3')

    SOURCE_FILENAMES = [
        'dataaz_reviews_df.pkl', 'logsinfo.log', 'logserrors.log',
        'datavocab.pkl'
    ]
    BUCKET_NAME = 'yummy-nlp'
    for SOURCE_FILENAME in SOURCE_FILENAMES
        logger.info(Uploading {} to S3.format(SOURCE_FILENAME))
        S3.upload_file(SOURCE_FILENAME, BUCKET_NAME, SOURCE_FILENAME)


def main()
    setup_logging()
    logger.info(mainstart-time ---  {}.format(
        datetime.datetime.now().time()))
    start_time = time.time()

    df = read_az_reviews()
    x, y = init_classification(df)
    vocab = init_transformation(x)
    x = vectorize(vocab, x)

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=101)

    nb = multinomial_naive_bayes(x_train, x_test, y_train, y_test)
    rfc = random_forest_classifier(x_train, x_test, y_train, y_test)
    svm = SVM(x_train, x_test, y_train, y_test)
    dt = decision_tree(x_train, x_test, y_train, y_test)
    knn = kNN(x_train, x_test, y_train, y_test)
    mlp = multilayer_perceptron(x_train, x_test, y_train, y_test)

    logger.info(Testing with Multilayer Perceptron)
    test_rating(df, mlp, 2, positive_rating, vocab)
    test_rating(df, mlp, 12, negative_rating, vocab)
    test_rating(df, mlp, 1, neutral_rating, vocab)

    upload_files_to_s3()
    logger.info(mainstart-time ---  {}.format(
        datetime.datetime.now().time()))
    logger.info(Elapsed {} seconds.format(round(time.time() - start_time,
                                                   4)))


if __name__ == '__main__'
    main()
