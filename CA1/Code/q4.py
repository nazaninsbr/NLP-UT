from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends
import numpy as np

def calculate_perplexity(model, test_text):
    perplexity = 0
    N = len(test_text)
    for word in test_text:
        N += 1
        unigram_prob = model.score(word[0])
        if unigram_prob == 0 or unigram_prob == 0.0:
            perplexity = perplexity + -10
        else:
            perplexity = perplexity + np.log(unigram_prob)
    perplexity = perplexity * 1/float(N)
    perplexity = np.power(2, -perplexity)
    return perplexity

def predict_the_class(best_model, news_item):
    min_perplexity, min_perplexity_label = None, None 
    for this_news_type in best_model.keys():
        p = calculate_perplexity(best_model[this_news_type], news_item)
        if min_perplexity == None or p < min_perplexity:
            min_perplexity = p 
            min_perplexity_label = this_news_type
    return min_perplexity_label

def run_q4_code(best_model, test_data_dict):
    test_data_dict = test_data_dict['UnknownLabel']
    with open('../Result.csv', 'w') as fp:
        fp.write('Filename,Class\n')
        for filename in test_data_dict.keys():
            this_text = test_data_dict[filename]
            this_text_arr = []
            for sent in this_text:
                this_text_arr.extend(sent)
            preped_data = list(ngrams(pad_both_ends(this_text_arr, n=1), 1))
            predicted_class = predict_the_class(best_model, preped_data)
            line_to_write = filename+','+predicted_class+'\n'
            fp.write(line_to_write)