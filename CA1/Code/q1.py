def perform_sentence_segmentation(data_dict):
    from parsivar import Tokenizer
    my_tokenizer = Tokenizer()

    return_value = {}
    for folder_name in data_dict.keys():
        return_value[folder_name] = {}
        for file_name in data_dict[folder_name].keys():
            tmp_text = data_dict[folder_name][file_name]
            token_text = my_tokenizer.tokenize_sentences(tmp_text)
            return_value[folder_name][file_name] = token_text
    
    return return_value

def perform_word_tokenization(data_dict):
    from parsivar import Tokenizer
    my_tokenizer = Tokenizer()

    return_value = {}
    for folder_name in data_dict.keys():
        return_value[folder_name] = {}
        for file_name in data_dict[folder_name].keys():
            this_sentences_words = []
            for sent_text in data_dict[folder_name][file_name]:
                token_text = my_tokenizer.tokenize_words(sent_text)
                this_sentences_words.append(token_text)
            return_value[folder_name][file_name] = this_sentences_words
    
    return return_value

def space_correct_data(data_dict):
    from parsivar import Normalizer
    my_normalizer = Normalizer(statistical_space_correction=True)
    return_value = {}
    for folder_name in data_dict.keys():
        return_value[folder_name] = {}
        for file_name in data_dict[folder_name].keys():
            tmp_text = data_dict[folder_name][file_name]
            normal_text = my_normalizer.normalize(tmp_text)
            return_value[folder_name][file_name] = normal_text
    return return_value

def hazm_normalize_data(data_dict):
    from hazm import Normalizer
    my_normalizer = Normalizer()

    return_value = {}
    for folder_name in data_dict.keys():
        return_value[folder_name] = {}
        for file_name in data_dict[folder_name].keys():
            tmp_text = data_dict[folder_name][file_name]
            normal_text = my_normalizer.normalize(tmp_text)
            return_value[folder_name][file_name] = normal_text
    
    return return_value

def pereditor_normalize_data(data_dict):
    import virastar
    pe = virastar.PersianEditor()

    return_value = {}
    for folder_name in data_dict.keys():
        return_value[folder_name] = {}
        for file_name in data_dict[folder_name].keys():
            tmp_text = data_dict[folder_name][file_name]
            normal_text = pe.cleanup(tmp_text)
            return_value[folder_name][file_name] = normal_text
    
    return return_value


def perform_word_lemmatization(data_dict):
    from hazm import Lemmatizer
    lemmatizer = Lemmatizer()

    return_value = {}
    for folder_name in data_dict.keys():
        return_value[folder_name] = {}
        for file_name in data_dict[folder_name].keys():
            this_files_words = []
            for sent_text in data_dict[folder_name][file_name]:
                this_sentences_words = []
                for word in sent_text:
                    lemma_word = lemmatizer.lemmatize(word)
                    this_sentences_words.append(lemma_word)
                this_files_words.append(this_sentences_words)
            return_value[folder_name][file_name] = this_files_words

    return return_value

def perform_word_stemming(data_dict):
    from hazm import Stemmer
    stemmer = Stemmer()

    return_value = {}
    for folder_name in data_dict.keys():
        return_value[folder_name] = {}
        for file_name in data_dict[folder_name].keys():
            this_files_words = []
            for sent_text in data_dict[folder_name][file_name]:
                this_sentences_words = []
                for word in sent_text:
                    lemma_word = stemmer.stem(word)
                    this_sentences_words.append(lemma_word)
                this_files_words.append(this_sentences_words)
            return_value[folder_name][file_name] = this_files_words

    return return_value


def run_q1_code(train_data):
    print('Politics, News 10022:\n', train_data['politic']['10022.txt'])
    normalized_data = hazm_normalize_data(train_data)
    per_editor_normalized_data = pereditor_normalize_data(normalized_data)
    parsivar_space_corrected_data = space_correct_data(per_editor_normalized_data)
    print('Normalized (Politics, News 10022):\n', parsivar_space_corrected_data['politic']['10022.txt'])
    sentence_segmented_data = perform_sentence_segmentation(parsivar_space_corrected_data)
    print('Sentence Segmented (Politics, News 10022):\n', sentence_segmented_data['politic']['10022.txt'])
    word_segmented_data = perform_word_tokenization(sentence_segmented_data)
    print('Word Segmented (Politics, News 10022):\n', word_segmented_data['politic']['10022.txt'])
    lemmatized_words = perform_word_lemmatization(word_segmented_data)
    print('Lemmatized (Politics, News 10022):\n', lemmatized_words['politic']['10022.txt'])
    # stemmed_words = perform_word_stemming(lemmatized_words)
    # print('Stemmed (Politics, News 10022):\n', stemmed_words['politic']['10022.txt'])

def clean_text(data_dict, do_lemmatization):
    normalized_data = hazm_normalize_data(data_dict)
    per_editor_normalized_data = pereditor_normalize_data(normalized_data)
    parsivar_space_corrected_data = space_correct_data(per_editor_normalized_data)
    sentence_segmented_data = perform_sentence_segmentation(parsivar_space_corrected_data)
    word_segmented_data = perform_word_tokenization(sentence_segmented_data)
    if do_lemmatization:
        lemmatized_words = perform_word_lemmatization(word_segmented_data)
        return lemmatized_words
    else:
        return word_segmented_data
    