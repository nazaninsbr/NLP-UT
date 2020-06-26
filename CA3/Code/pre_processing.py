import nltk
import re
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def remove_punctuation(text):
    return re.sub(r'[^(a-zA-Z)\s]','', text)

def get_word_tokens(raw):
    return nltk.word_tokenize(raw)
    # return raw.split(' ')

def normalize_words(tokens):
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens

def preform_pre_processing(raw):
    # txt = remove_punctuation(raw)
    words = get_word_tokens(raw)
    words = normalize_words(words)
    return words