import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords as nltk_stopwords
import re 

def remove_punctuation(text):
    return re.sub(r'[^(a-zA-Z)\s]','', text)

def get_word_tokens(raw):
    return nltk.word_tokenize(raw)

def normalize_words(tokens):
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens

def remove_stop_words(tokens):
    return [word for word in tokens if not word in nltk_stopwords.words("english")]

def lemmatize_words(tokens):
    res = []
    lemma = WordNetLemmatizer()
    for token, tag in pos_tag(tokens):
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        token = lemma.lemmatize(token, pos)
        res.append(token)
    return res

def clean_text(text):
    text = remove_punctuation(text)
    tokens = get_word_tokens(text)
    tokens = normalize_words(tokens)
    tokens = remove_stop_words(tokens)
    tokens = lemmatize_words(tokens)
    return tokens

def clean_data(all_X):
    count = 0
    cleaned_X = []
    for item in all_X:
        count += 1
        if count%500==1:
            print('count', count)
        cleaned_item = clean_text(item)
        cleaned_X.append(cleaned_item)
    return cleaned_X
