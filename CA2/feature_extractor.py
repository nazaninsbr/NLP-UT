import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def get_char_len(text):
    return len(text)

def get_number_of_exclamation_marks(text):
    return text.count('!')

def get_number_of_question_marks(text):
    return text.count('?')

def get_number_of_uppercase_letters(text):
    all_words = nltk.word_tokenize(text)
    num_of_uppercase = 0 
    for w in all_words:
        if w.isupper():
            num_of_uppercase += 1
    return num_of_uppercase

def get_number_of_adjectives(words):
    adj_list, num_adj = [], 0
    try:
        pos = nltk.pos_tag(words)
        for item in pos:
            if item[1] == 'JJ':
                num_adj += 1
                adj_list.append(item[0])
    except Exception:
        pass
    return num_adj, adj_list

def get_number_of_pos_words(words):
    pos_words = []
    with open('pos_words.txt', 'r') as fp:
        c = fp.read()
        pos_words = c.split('\n')

    num_pos = 0
    for w in words:
        if w in pos_words:
            num_pos += 1
    
    return num_pos

def get_number_of_neg_words(words):
    neg_words = []
    with open('neg_words.txt', 'r') as fp:
        c = fp.read()
        neg_words = c.split('\n')

    num_neg = 0
    for w in words:
        if w in neg_words:
            num_neg += 1
    
    return num_neg

def calculate_bag_of_words(all_data):
    all_texts = [' '.join(item) for item in all_data]
    vectorizer = CountVectorizer()
    vectorizer.fit(all_texts)
    vectors = vectorizer.transform(all_texts)
    return vectors.toarray()

def calculate_tfidf(all_data):
    all_texts = [' '.join(item) for item in all_data]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)
    vectors = vectorizer.transform(all_texts)
    return vectors.toarray()

def get_number_of_pronouns(text):
    lowered_text = text.lower()
    pronount_counts = lowered_text.count(' you ')+lowered_text.count(' me ')+lowered_text.count(' i ')+lowered_text.count(" you're ")+lowered_text.count(" i'm ")+lowered_text.count(" i've ")+lowered_text.count(" you've ")
    return pronount_counts

def extract_features(cleaned_X, actual_X, no_bag_of_words=False):
    number_of_data_points = len(cleaned_X)
    X_with_features = []

    bow_output = calculate_bag_of_words(cleaned_X)

    for ind in range(number_of_data_points):

        if ind%500 == 1:
            print('text number', ind)

        char_len = get_char_len(actual_X[ind])
        number_of_exl_mark = get_number_of_exclamation_marks(actual_X[ind])
        number_of_uppercase_letters = get_number_of_uppercase_letters(actual_X[ind])
        number_of_q_mark = get_number_of_question_marks(actual_X[ind])
        number_of_positive_words = get_number_of_pos_words(cleaned_X[ind])
        number_of_negative_words = get_number_of_neg_words(cleaned_X[ind])
        number_of_adjectives, adj_list = get_number_of_adjectives(cleaned_X[ind])

        number_of_positive_adjectives = get_number_of_pos_words(adj_list)
        number_of_negative_adjectives = get_number_of_neg_words(adj_list)

        number_of_pronouns = get_number_of_pronouns(actual_X[ind])

        if no_bag_of_words == True:
            this_items_features = []
        else:
            this_items_features = bow_output[ind].tolist()
        calculated_features = [
            char_len, 
            number_of_exl_mark, 
            number_of_uppercase_letters, 
            number_of_q_mark, 
            number_of_positive_words, 
            number_of_negative_words, 
            number_of_adjectives, 
            number_of_positive_adjectives, 
            number_of_negative_adjectives, 
            number_of_pronouns
        ]
        this_items_features.extend(calculated_features)

        X_with_features.append(this_items_features)


    return X_with_features


def extract_features_TFIDF(cleaned_X, actual_X):
    number_of_data_points = len(cleaned_X)
    X_with_features = []

    tfidf_output = calculate_tfidf(cleaned_X)

    for ind in range(number_of_data_points):

        if ind%500 == 1:
            print('text number', ind)

        char_len = get_char_len(actual_X[ind])
        number_of_exl_mark = get_number_of_exclamation_marks(actual_X[ind])
        number_of_uppercase_letters = get_number_of_uppercase_letters(actual_X[ind])
        number_of_q_mark = get_number_of_question_marks(actual_X[ind])
        number_of_positive_words = get_number_of_pos_words(cleaned_X[ind])
        number_of_negative_words = get_number_of_neg_words(cleaned_X[ind])
        number_of_adjectives, adj_list = get_number_of_adjectives(cleaned_X[ind])
        number_of_positive_adjectives = get_number_of_pos_words(adj_list)
        number_of_negative_adjectives = get_number_of_neg_words(adj_list)

        this_items_features = tfidf_output[ind].tolist()
        calculated_features = [
            char_len, 
            number_of_exl_mark, 
            number_of_uppercase_letters, 
            number_of_q_mark, 
            number_of_positive_words, 
            number_of_negative_words, 
            number_of_adjectives, 
            number_of_positive_adjectives, 
            number_of_negative_adjectives
        ]
        this_items_features.extend(calculated_features)

        X_with_features.append(this_items_features)

    return X_with_features

def perform_bag_of_words(cleaned_X):
    bow_output = calculate_bag_of_words(cleaned_X)
    return bow_output