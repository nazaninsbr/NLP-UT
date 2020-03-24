from data_downloader import get_dataset
from data_cleaner import clean_data
from feature_extractor import extract_features, perform_bag_of_words, extract_features_TFIDF
from model_training import train_the_model

def model_with_all_preprocessing_and_features(no_bag_of_words=False):
    print('## Download and load the data')
    dataset = get_dataset()
    print('## Clean the data')
    cleaned_X = clean_data(dataset['X'])
    print('## Feature Extraction')
    all_features_X = extract_features(cleaned_X, dataset['X'], no_bag_of_words=no_bag_of_words)
    print('## Training and Evaluation')
    train_the_model(all_features_X, dataset['Y'])

def model_with_all_preprocessing_and_features_TFIDF():
    print('## Download and load the data')
    dataset = get_dataset()
    print('## Clean the data')
    cleaned_X = clean_data(dataset['X'])
    print('## Feature Extraction')
    all_features_X = extract_features_TFIDF(cleaned_X, dataset['X'])
    print('## Training and Evaluation')
    train_the_model(all_features_X, dataset['Y'])

def model_with_no_preprocessing_or_features():
    print('## Download and load the data')
    dataset = get_dataset()
    print('## Only separate using spaces')
    cleaned_X = [val.split(' ') for val in dataset['X']]
    print('## Only perform BoW')
    all_features_X = perform_bag_of_words(cleaned_X)
    print('## Training and Evaluation')
    train_the_model(all_features_X, dataset['Y'])

def model_with_only_features():
    print('## Download and load the data')
    dataset = get_dataset()
    print('## Only separate using spaces')
    cleaned_X = [val.split(' ') for val in dataset['X']]
    print('## Feature Extraction')
    all_features_X = extract_features(cleaned_X, dataset['X'])
    print('## Training and Evaluation')
    train_the_model(all_features_X, dataset['Y'])

def model_with_only_preprocessing():
    print('## Download and load the data')
    dataset = get_dataset()
    print('## Clean the data')
    cleaned_X = clean_data(dataset['X'])
    print('## Only perform BoW')
    all_features_X = perform_bag_of_words(cleaned_X)
    print('## Training and Evaluation')
    train_the_model(all_features_X, dataset['Y'])


print('**************** no preprocessing or features')
model_with_no_preprocessing_or_features()
print('**************** only preprocessing, no features')
model_with_only_preprocessing()
print('**************** no preprocessing, only features')
model_with_only_features()
print('**************** preprocessing and features')
model_with_all_preprocessing_and_features()
print('**************** preprocessing and features without BoW')
model_with_all_preprocessing_and_features(no_bag_of_words=True)
print('**************** preprocessing and features with TFIDF')
model_with_all_preprocessing_and_features_TFIDF()