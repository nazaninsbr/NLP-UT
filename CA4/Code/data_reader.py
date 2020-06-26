import pandas as pd
import numpy as np 
from keras.preprocessing.sequence import pad_sequences

def read_pos_list():
    file_path = '../data/pos_list.xlsx'
    df = pd.read_excel(file_path)
    pos_values = df['POS_list'].values
    this_tag_number = 1
    pos_tag_mapping = {}
    for val in pos_values:
        if not val in pos_tag_mapping.keys():
            pos_tag_mapping[val] = this_tag_number
            this_tag_number += 1
    return pos_tag_mapping

def convert_string_to_int(X_train, X_test, word_number_mapping):
    train, test = [], []
    maxlen = None 
    for sent in X_train:
        this_sent = []
        if maxlen == None or len(sent)>maxlen:
            maxlen = len(sent)
        for word in sent:
            this_sent.append(word_number_mapping[word])
        train.append(this_sent)

    for sent in X_test:
        this_sent = []
        if maxlen == None or len(sent)>maxlen:
            maxlen = len(sent)
        for word in sent:
            if word in word_number_mapping.keys():
                this_sent.append(word_number_mapping[word])
            else:
                this_sent.append(word_number_mapping['OOV'])
        test.append(this_sent)
    train, test = pad_sequences(train, maxlen=maxlen), pad_sequences(test, maxlen=maxlen)
    return train, test, maxlen

def convert_labels(y_train, y_test, pos_tag_mapping, maxlen):
    train, test = [], []
    for sent in y_train:
        this_sent = []
        for word in sent:
            this_sent.append(pos_tag_mapping[word])
        train.append(this_sent)

    for sent in y_test:
        this_sent = []
        for word in sent:
            this_sent.append(pos_tag_mapping[word])
        test.append(this_sent)

    train, test = pad_sequences(train, maxlen=maxlen), pad_sequences(test, maxlen=maxlen)
    return train, test


def to_categorical(y_labels, categories):
    categorical_labels = []
    for s in y_labels:
        this_item = []
        for item in s:
            this_item.append(np.zeros(categories))
            this_item[-1][item] = 1.0
        categorical_labels.append(this_item)
    return np.array(categorical_labels)

def read_rnn_test_and_train():
    train_file_path = '../data/Rnn_train.xlsx'
    test_file_path = '../data/Rnn_test.xlsx'

    pos_tag_mapping = read_pos_list() 

    word_number_mapping = {'OOV': 1} 
    this_word_number = 2

    X_train, y_train = [], []
    X_test, y_test = [], []
    this_item_x, this_item_y = [], []

    train_df = pd.read_excel(train_file_path)
    for _, row in train_df.iterrows():
        if 'Sentence' in str(row['Sentence #']):
            X_train.append(this_item_x)
            y_train.append(this_item_y)
            this_item_x, this_item_y = [], []
        this_word = str(row['Word']).lower()
        this_item_x.append(this_word)
        if not this_word in word_number_mapping.keys():
            word_number_mapping[this_word] = this_word_number
            this_word_number += 1
        this_item_y.append(row['POS'])
    X_train.append(this_item_x)
    y_train.append(this_item_y)
   
    test_df = pd.read_excel(test_file_path, header=None)
    this_item_x, this_item_y = [], []
    for _, row in test_df.iterrows():
        if 'Sentence' in str(row[0]):
            X_test.append(this_item_x)
            y_test.append(this_item_y)
            this_item_x, this_item_y = [], []
        this_word = str(row[1]).lower()
        this_item_x.append(this_word)
        this_item_y.append(row[2])
    X_test.append(this_item_x)
    y_test.append(this_item_y)
    
    X_train, y_train, X_test, y_test = X_train[1:], y_train[1:], X_test[1:], y_test[1:]

    X_train, X_test, maxlen = convert_string_to_int(X_train, X_test, word_number_mapping)
    y_train, y_test = convert_labels(y_train, y_test, pos_tag_mapping, maxlen)

    y_train = to_categorical(y_train, categories=43) 
    y_test = to_categorical(y_test, categories=43) 

    return X_train, y_train, X_test, y_test

def read_viterbi_test_and_train():
    train_file_path = '../data/viterbi_train.xlsx'
    test_file_path = '../data/viterbi_test.xlsx'

    word_number_mapping = {'OOV': 1} 
    this_word_number = 2

    X_train, y_train = [], []
    X_test, y_test = [], []
    this_item_x, this_item_y = [], []

    train_df = pd.read_excel(train_file_path, header=None)
    for _, row in train_df.iterrows():
        if 'Sentence' in str(row[0]):
            X_train.append(this_item_x)
            y_train.append(this_item_y)
            this_item_x, this_item_y = [], []
        this_word = str(row[1]).lower()
        this_item_x.append(this_word)
        if not this_word in word_number_mapping.keys():
            word_number_mapping[this_word] = this_word_number
            this_word_number += 1
        this_item_y.append(row[2])
    X_train.append(this_item_x)
    y_train.append(this_item_y)
   
    test_df = pd.read_excel(test_file_path, header=None)
    this_item_x, this_item_y = [], []
    for _, row in test_df.iterrows():
        if 'Sentence' in str(row[0]):
            X_test.append(this_item_x)
            y_test.append(this_item_y)
            this_item_x, this_item_y = [], []
        this_word = str(row[1]).lower()
        this_item_x.append(this_word)
        this_item_y.append(row[2])
    X_test.append(this_item_x)
    y_test.append(this_item_y)

    X_train, y_train, X_test, y_test = X_train[1:], y_train[1:], X_test[1:], y_test[1:]

    return X_train, y_train, X_test, y_test, word_number_mapping