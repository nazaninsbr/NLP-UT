import keras 
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, TimeDistributed, Activation, Embedding
import keras.backend as K
from sklearn.metrics import classification_report
import numpy as np 
from data_reader import read_pos_list

class RNN:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)

    def create_rnn_model(self):
        input_len = 104
        output_len = 43
        number_of_unique_words = 21124

        model = Sequential()
        model.add(InputLayer(input_shape=(input_len, )))
        model.add(Embedding(number_of_unique_words, 128))
        model.add(LSTM(256, return_sequences=True))
        model.add(TimeDistributed(Dense(output_len)))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
        print(model.summary())
        return model

    def convert_categorical_to_label(self, y_true_labels, y_predicted):
        pos_tag_mapping = read_pos_list()
        tag_number_mapping = {pos_tag_mapping[k]:k for k in pos_tag_mapping.keys()}
        tag_number_mapping[0] = 'padding'
        
        y_true, y_pred = [], []

        for sentence in y_true_labels:
            for label in sentence:
                this_number = np.argmax(label)
                this_tag = tag_number_mapping[this_number]
                y_true.append(this_tag)

        for sentence in y_predicted:
            for label in sentence:
                this_number = np.argmax(label)
                this_tag = tag_number_mapping[this_number]
                y_pred.append(this_tag)

        y_to_write = []
        for y_id in range(len(y_true)):
            if not y_true[y_id] == 'padding':
                y_to_write.append(y_pred[y_id])

        return y_true, y_pred, y_to_write

    def train_the_model(self, model):
        print(self.X_train.shape)
        model.fit(self.X_train, self.y_train, epochs=25, batch_size=128)
        scores = model.evaluate(self.X_test, self.y_test)
        print('Accuracy: (%)', scores[1] * 100)

        y_predicted = model.predict(self.X_test)
        y_true, y_pred, y_to_write = self.convert_categorical_to_label(self.y_test, y_predicted)
        print(classification_report(y_true, y_pred))

        with open('rnn_predicted_labels.txt', 'w') as fp:
            for this_label in y_to_write:
                fp.write(this_label+'\n')

    def run(self):
        model = self.create_rnn_model()
        self.train_the_model(model)