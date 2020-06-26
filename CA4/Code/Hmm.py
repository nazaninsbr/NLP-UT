import numpy as np 
from data_reader import read_pos_list
from sklearn.metrics import classification_report

class HMM:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def viterbi_find_best_sequence(self, sentence, emission_probabilities, transition_probabilities, pos_tag_mapping, word_number_mapping):
        viterbi_prev_step, viterbi_this_step = {t:{'prob': 0, 'steps':[]} for t in pos_tag_mapping.keys()}, {t:{'prob': 0, 'steps':[]} for t in pos_tag_mapping.keys()}
        viterbi_prev_step['<S>']['prob'] = 1
        sentence = ['<S>'] + sentence + ['</S>']

        for word_ind in range(1, len(sentence)):
            this_word = sentence[word_ind]
            if this_word in word_number_mapping.keys():
                word_number = word_number_mapping[this_word] - 1
            else:
                word_number = word_number_mapping['OOV'] - 1
            for this_tag in pos_tag_mapping.keys():
                if this_tag == '<S>':
                    continue
                this_tag_number =  pos_tag_mapping[this_tag] -1 
                p_word_tag = emission_probabilities[word_number][this_tag_number]
                best_prev_tag, best_prev_tag_prob = None, None 
                for prev_tag in viterbi_prev_step.keys():
                    prev_tag_number = pos_tag_mapping[prev_tag] -1 
                    p_trans = transition_probabilities[prev_tag_number][this_tag_number]
                    prob_so_far = viterbi_prev_step[prev_tag]['prob'] * p_trans * p_word_tag
                    if best_prev_tag_prob == None or best_prev_tag_prob<prob_so_far:
                        best_prev_tag_prob = prob_so_far
                        best_prev_tag = prev_tag
                viterbi_this_step[this_tag]['prob'] = best_prev_tag_prob
                viterbi_this_step[this_tag]['steps'] = viterbi_prev_step[best_prev_tag]['steps'] + [this_tag]

            viterbi_prev_step = viterbi_this_step
            viterbi_this_step = {t:{'prob': 0, 'steps':[]} for t in pos_tag_mapping.keys()}
        
        prediction = viterbi_prev_step['</S>']['steps'][:-1]
        return prediction


    def run(self, word_number_mapping):
        pos_tag_mapping = read_pos_list()
        pos_tag_mapping['<S>'] = len(pos_tag_mapping) + 1
        pos_tag_mapping['</S>'] = len(pos_tag_mapping) + 1
        word_part_of_speech_table = np.ones((len(word_number_mapping), len(pos_tag_mapping)))

        for sentence_id in range(len(self.X_train)):
            for word_id in range(len(self.X_train[sentence_id])):
                this_word = self.X_train[sentence_id][word_id]
                this_word_number = word_number_mapping[this_word] - 1
                this_label = self.y_train[sentence_id][word_id]
                this_label_number = pos_tag_mapping[this_label] - 1
                word_part_of_speech_table[this_word_number][this_label_number] += 1

        emission_probabilities = word_part_of_speech_table/word_part_of_speech_table.sum(axis=1, keepdims=True)
        transition_probabilities = np.ones((len(pos_tag_mapping), len(pos_tag_mapping))) 

        for sentence_id in range(len(self.X_train)):
            for word_id in range(len(self.X_train[sentence_id])):
                this_label = self.y_train[sentence_id][word_id]
                this_label_number = pos_tag_mapping[this_label] - 1
                if word_id == 0:
                    last_label = '<S>'
                    last_label_number = pos_tag_mapping[last_label] - 1
                    transition_probabilities[last_label_number][this_label_number] += 1
                if word_id == len(self.X_train[sentence_id])-1:
                    last_label = self.y_train[sentence_id][word_id-1]
                    last_label_number = pos_tag_mapping[last_label] - 1
                    transition_probabilities[last_label_number][this_label_number] += 1

                    next_label = '</S>'
                    next_label_number = pos_tag_mapping[next_label] - 1
                    transition_probabilities[this_label_number][next_label_number] += 1
                else:
                    last_label = self.y_train[sentence_id][word_id-1]
                    last_label_number = pos_tag_mapping[last_label] - 1
                    transition_probabilities[last_label_number][this_label_number] += 1
        
        y_predictions = []
        y_true = []
        for sentence_id in range(len(self.X_test)):
            sentence = self.X_test[sentence_id]
            best_tag_sequence = self.viterbi_find_best_sequence(sentence, emission_probabilities, transition_probabilities, pos_tag_mapping, word_number_mapping)
            y_predictions.extend(best_tag_sequence)
            y_true.extend(self.y_test[sentence_id])

        print(classification_report(y_true, y_predictions))
        with open('hmm_predicted_labels.txt', 'w') as fp:
            for this_label in y_predictions:
                fp.write(this_label+'\n')