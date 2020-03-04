from nltk.util import ngrams
from nltk.util import pad_sequence
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
import random
from nltk.lm import MLE

def divide_test_and_train_data(data_dict, train_percentage):
    train_X, test_X = {}, {}
    for folder_name in data_dict.keys():
        train_X[folder_name] = []
        test_X[folder_name] = []
        for file_name in data_dict[folder_name].keys():
            if random.uniform(0, 1) >= 1 - train_percentage:
                train_X[folder_name].append(data_dict[folder_name][file_name])
            else:
                test_X[folder_name].append(data_dict[folder_name][file_name])
    return train_X, test_X 

def calculate_word_ngrams(data):
    text_bigrams, text_unigrams = {}, {}
    for news_type in data.keys():
        all_news_type_texts = []
        for news in data[news_type]:
            all_news_texts = []
            for sent in news:
                all_news_texts.extend(sent)
            all_news_type_texts.append(all_news_texts)
        train_bi, vocab_bi = padded_everygram_pipeline(2, all_news_type_texts)
        text_bigrams[news_type] = {'train': train_bi, 'vocab': vocab_bi}
        train_uni, vocab_uni = padded_everygram_pipeline(1, all_news_type_texts)
        text_unigrams[news_type] = {'train': train_uni, 'vocab': vocab_uni}
    return text_unigrams, text_bigrams

def calculate_characters_ngrams(data):
    text_bigrams, text_unigrams = {}, {}
    for news_type in data.keys():
        all_news_type_texts = []
        for news in data[news_type]:
            all_news_texts = []
            for sent in news:
                for word in sent:
                    all_chars = [c for c in word]+[' ']
                    all_news_texts.extend(all_chars)
            all_news_type_texts.append(all_news_texts)
        train_bi, vocab_bi = padded_everygram_pipeline(2, all_news_type_texts)
        text_bigrams[news_type] = {'train': train_bi, 'vocab': vocab_bi}
        train_uni, vocab_uni = padded_everygram_pipeline(1, all_news_type_texts)
        text_unigrams[news_type] = {'train': train_uni, 'vocab': vocab_uni}
    return text_unigrams, text_bigrams

def prep_test_data(data):
    preped_test_data = {news_type: {'word': {'unigram': [], 'bigram': []}, 'character': {'unigram': [], 'bigram': []}} for news_type in data.keys()}

    for news_type in data.keys():
        for news in data[news_type]:
            this_news_sentences = []
            this_news_characters = []
            for sent in news:
                this_news_sentences.extend(sent)
                for word in sent:
                    all_chars = [c for c in word]+[' ']
                this_news_characters.extend(all_chars)

            preped_test_data[news_type]['word']['bigram'].append(list(ngrams(pad_both_ends(this_news_sentences, n=2), 2)))
            preped_test_data[news_type]['word']['unigram'].append(list(ngrams(pad_both_ends(this_news_sentences, n=1), 1)))

            preped_test_data[news_type]['character']['bigram'].append(list(ngrams(pad_both_ends(this_news_characters, n=2), 2)))
            preped_test_data[news_type]['character']['unigram'].append(list(ngrams(pad_both_ends(this_news_characters, n=1), 1)))
    
    return preped_test_data


def create_all_language_models(text_bigram_words, text_unigram_words, text_bigram_characters, text_unigram_characters):
    all_models = {'Word': {'Unigram': {}, 'Bigram': {}}, 'Character': {'Unigram': {}, 'Bigram': {}}}

    for news_type in text_bigram_words.keys():
        this_model = MLE(2)
        this_model.fit(text_bigram_words[news_type]['train'], text_bigram_words[news_type]['vocab'])
        all_models['Word']['Bigram'][news_type] = this_model 

    for news_type in text_bigram_characters.keys():
        this_model = MLE(2)
        this_model.fit(text_bigram_characters[news_type]['train'], text_bigram_characters[news_type]['vocab'])
        all_models['Character']['Bigram'][news_type] = this_model 

    for news_type in text_unigram_words.keys():
        this_model = MLE(1)
        this_model.fit(text_unigram_words[news_type]['train'], text_unigram_words[news_type]['vocab'])
        all_models['Word']['Unigram'][news_type] = this_model 

    for news_type in text_unigram_characters.keys():
        this_model = MLE(1)
        this_model.fit(text_unigram_characters[news_type]['train'], text_unigram_characters[news_type]['vocab'])
        all_models['Character']['Unigram'][news_type] = this_model 

    return all_models

def run_q2_code(data_dict):
    train_X_Y, test_X_Y = divide_test_and_train_data(data_dict, train_percentage=0.8)

    print('Train Count:', sum([len(train_X_Y[k]) for k in train_X_Y.keys()]))
    print('Train Examples in each Category:', {k: len(train_X_Y[k]) for k in train_X_Y.keys()})
    print('Test Count:', sum([len(test_X_Y[k]) for k in test_X_Y.keys()]))

    text_bigram_words, text_unigram_words = calculate_word_ngrams(train_X_Y)
    text_bigram_characters, text_unigram_characters = calculate_characters_ngrams(train_X_Y)

    all_models = create_all_language_models(text_bigram_words, text_unigram_words, text_bigram_characters, text_unigram_characters)

    print('Checking the word-bigram model:')
    print('# Vocab:', len(all_models['Word']['Bigram']['politic'].vocab))
    preped_test_X_Y = prep_test_data(test_X_Y)
    print(preped_test_X_Y['politic']['word']['bigram'][0])

def train_lang_models_and_prep_test_data(data_dict, no_label_test_dict=False):
    if no_label_test_dict == False:
        train_X_Y, test_X_Y = divide_test_and_train_data(data_dict, train_percentage=0.8)
        text_bigram_words, text_unigram_words = calculate_word_ngrams(train_X_Y)
        text_bigram_characters, text_unigram_characters = calculate_characters_ngrams(train_X_Y)
        all_models = create_all_language_models(text_bigram_words, text_unigram_words, text_bigram_characters, text_unigram_characters)
        preped_test_X_Y = prep_test_data(test_X_Y)
    else:
        train_X_Y, test_X_Y = divide_test_and_train_data(data_dict, train_percentage=1)
        text_bigram_words, text_unigram_words = calculate_word_ngrams(train_X_Y)
        text_bigram_characters, text_unigram_characters = calculate_characters_ngrams(train_X_Y)
        all_models = create_all_language_models(text_bigram_words, text_unigram_words, text_bigram_characters, text_unigram_characters)
        preped_test_X_Y = None
    return all_models, preped_test_X_Y
