from data_reader import read_pos_list, read_rnn_test_and_train, read_viterbi_test_and_train
from Rnn import RNN
from Hmm import HMM

# X_train, y_train, X_test, y_test = read_rnn_test_and_train()
# print('X & Y Example:', (X_train[0], y_train[0]))
# print('### RNN ###')
# RNN(X_train, y_train, X_test, y_test).run()
print('### HMM ###')
X_train, y_train, X_test, y_test, word_number_mapping = read_viterbi_test_and_train()
HMM(X_train, y_train, X_test, y_test).run(word_number_mapping)