from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np 
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def calculate_perplexity(model, test_text, bi_or_uni):
    perplexity = 0
    if bi_or_uni == 'unigram':
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
    if bi_or_uni == 'biogram':
        N = len(test_text) - 1 
        for word in test_text:
            bi_prob = model.score(word[1], word[0])
            if bi_prob == 0 or bi_prob == 0.0:
                perplexity = perplexity + -10
            else:
                perplexity = perplexity + np.log(bi_prob)
        perplexity = perplexity * 1/float(N)
        perplexity = np.power(2, -perplexity)
    return perplexity

def evaluate_this_model(these_models, word_or_char, uni_or_bigram, test_data):
    true_labels, predicted_labels = [], []
    for news_type in test_data.keys():
        for news_item in test_data[news_type][word_or_char][uni_or_bigram]:
            min_perplexity, min_perplexity_label = None, None 
            for this_news_type in these_models.keys():
                try:
                    p = calculate_perplexity(these_models[this_news_type], news_item, uni_or_bigram)
                    if min_perplexity == None or p < min_perplexity:
                        min_perplexity = p 
                        min_perplexity_label = this_news_type
                except Exception as e:
                    print(e)
                    exit()
            predicted_labels.append(min_perplexity_label)
            true_labels.append(news_type)

    print('Accuracy:', accuracy_score(true_labels, predicted_labels))
    print('Recall (weighted):', recall_score(true_labels, predicted_labels, average='weighted'))
    print('F1 (macro):', f1_score(true_labels, predicted_labels, average='macro'))
    print('F1 (micro):', f1_score(true_labels, predicted_labels, average='micro'))
    print('F1 (weighted):', f1_score(true_labels, predicted_labels, average='weighted'))
    labels = [k for k in test_data.keys()]
    print('Report:', classification_report(true_labels, predicted_labels, target_names=labels))
    print('Confusion Matrix:')
    cm_labels = np.unique(true_labels)
    conf_array = confusion_matrix(true_labels, predicted_labels, cm_labels)
    print(conf_array)
    df_cm = pd.DataFrame(conf_array, [k for k in test_data.keys()], [k for k in test_data.keys()])
    print(df_cm)
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    plt.show()

def run_q3_code(all_models, preped_test_X_Y):
    print('|> Testing the Word Unigram Model')
    these_models = all_models['Word']['Unigram']
    evaluate_this_model(these_models, 'word', 'unigram', preped_test_X_Y)

    print('|> Testing the Word Bigram Model')
    these_models = all_models['Word']['Bigram']
    evaluate_this_model(these_models, 'word', 'bigram', preped_test_X_Y)

    print('|> Testing the Character Unigram Model')
    these_models = all_models['Character']['Unigram']
    evaluate_this_model(these_models, 'character', 'unigram', preped_test_X_Y)

    print('|> Testing the Character Bigram Model')
    these_models = all_models['Character']['Bigram']
    evaluate_this_model(these_models, 'character', 'bigram', preped_test_X_Y)
    
    
    