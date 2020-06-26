import numpy as np 
from pre_processing import similar as similarity_between_strings
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
import matplotlib.pyplot as plt 
from keras import backend as K
import tensorflow as tf
from keras import optimizers

class calculate_perplexity(tf.keras.callbacks.Callback):
	def __init__(self, train_X, train_Y):
		self.train_data = (train_X, train_Y)
		self.train_per = []
		self.test_per = []
	def on_epoch_end(self, epoch, logs={}):
		predicted_train = self.model.predict(self.train_data[0])
		train_perplexity = K.exp(K.mean(K.categorical_crossentropy( tf.convert_to_tensor(self.train_data[1]),  tf.convert_to_tensor(predicted_train))))
		with tf.Session() as sess: 
			self.train_per.append(sess.run(train_perplexity))
		predicted_test = self.model.predict(self.validation_data[0])
		test_perplexity = K.exp(K.mean(K.categorical_crossentropy( tf.convert_to_tensor(self.validation_data[1]),  tf.convert_to_tensor(predicted_test))))
		with tf.Session() as sess: 
			self.test_per.append(sess.run(test_perplexity))

	def plot_per(self):
		plt.plot(self.train_per)
		plt.plot(self.test_per)
		plt.title('model perplexity')
		plt.ylabel('perplexity')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()


class NeuralNetwork:
	def __init__(self, number_of_hidden_layer_neurons, number_of_context_words, learning_rate, use_glove, vocab_size, projection_size, train_data,test_data, word_to_index):
		self.number_of_hidden_layer_neurons = number_of_hidden_layer_neurons
		self.number_of_context_words = number_of_context_words
		self.learning_rate = learning_rate
		self.use_glove = use_glove
		self.vocab_size = vocab_size
		self.projection_size = projection_size

		# data 
		self.train_data = train_data
		self.test_data = test_data
		
		self.train_data_X = [] 
		self.train_data_Y = []
		self.test_data_X = [] 
		self.test_data_Y = [] 

		self.word_to_index = word_to_index
		self.index_to_word = {word_to_index[k]:k for k in word_to_index.keys()}

	def set_parameters(self, number_of_hidden_layer_neurons, number_of_context_words, learning_rate):
		self.number_of_hidden_layer_neurons = number_of_hidden_layer_neurons
		self.number_of_context_words = number_of_context_words
		self.learning_rate = learning_rate
	
	def find_best_match(self, this_w):
		all_words = [w for w in self.use_glove.keys()]
		sim_scores = [similarity_between_strings(this_w, w) for w in all_words]
		max_sim_value = sim_scores.index(min(sim_scores))
		return all_words[max_sim_value]

	def get_vector_of_word_results(self, curr_word, context_words):
		if self.use_glove == None:
			context_word_vec = np.zeros((self.number_of_context_words, self.vocab_size))
			curr_word_vec = np.zeros((1, self.vocab_size))
			for cont_word_ind in range(len(context_words)):
				try:
					context_word_vec[cont_word_ind][self.word_to_index[context_words[cont_word_ind]]] = 1
				except Exception:
					context_word_vec[cont_word_ind][self.word_to_index['<UNK>']] = 1
				
			try:
				curr_word_vec[0][self.word_to_index[curr_word]] = 1
			except Exception:
				curr_word_vec[0][self.word_to_index['<UNK>']] = 1
		else:
			context_word_vec = np.zeros((self.number_of_context_words, self.projection_size))
			curr_word_vec = np.zeros(self.vocab_size)
			try:
				curr_word_vec[self.word_to_index[curr_word]] = 1
			except Exception:
				curr_word_vec[self.word_to_index['<UNK>']] = 1
			for cont_word_ind in range(len(context_words)):
				try:
					context_word_vec[cont_word_ind] = self.use_glove[context_words[cont_word_ind]]
				except Exception:
					best_match = self.find_best_match(context_words[cont_word_ind])
					context_word_vec[cont_word_ind] = self.use_glove[best_match]
		context_word_vec = context_word_vec.flatten()
		return context_word_vec, curr_word_vec

	def prep_data_for_test_and_train(self):
		for curr_word_index in range(len(self.train_data)):
			print(curr_word_index)
			curr_word = self.train_data[curr_word_index]
			if curr_word_index >= self.number_of_context_words:
				context_words = self.train_data[curr_word_index-self.number_of_context_words:curr_word_index]
			else:
				context_words = self.train_data[:curr_word_index]
				while len(context_words) < self.number_of_context_words:
					context_words = ['<S>']+context_words
			context_words_vec, curr_word_vec = self.get_vector_of_word_results(curr_word, context_words)
			self.train_data_X.append(context_words_vec)
			self.train_data_Y.append(curr_word_vec)
		
		self.train_data_X = np.array(self.train_data_X).astype(np.float32)
		self.train_data_Y = np.array(self.train_data_Y).astype(np.float32)

		for curr_word_index in range(len(self.test_data)):
			print(curr_word_index)
			curr_word = self.test_data[curr_word_index]
			if curr_word_index >= self.number_of_context_words:
				context_words = self.test_data[curr_word_index-self.number_of_context_words:curr_word_index]
			else:
				context_words = self.test_data[:curr_word_index]
				while len(context_words) < self.number_of_context_words:
					context_words = ['<S>']+context_words
			context_words_vec, curr_word_vec = self.get_vector_of_word_results(curr_word, context_words)
			self.test_data_X.append(context_words_vec)
			self.test_data_Y.append(curr_word_vec)

		self.test_data_X = np.array(self.test_data_X).astype(np.float32)
		self.test_data_Y = np.array(self.test_data_Y).astype(np.float32)

	def plot_accuracy_changes(self, history):
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()

	def train_keras_model(self, number_of_epochs):
		model = Sequential([
			Dense(self.number_of_context_words*self.projection_size, input_shape=(self.number_of_context_words*self.projection_size,)),
			Activation('linear'),
			Dense(self.number_of_hidden_layer_neurons),
			Activation('sigmoid'),
			Dense(self.vocab_size),
			Activation('softmax')
		])
		print(model.summary())
		sgd = optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(optimizer=sgd,
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
		cp = calculate_perplexity(self.train_data_X, self.train_data_Y)
		history = model.fit(self.train_data_X, self.train_data_Y, epochs=number_of_epochs, batch_size=256, validation_data=(self.test_data_X, self.test_data_Y), callbacks=[cp])
		self.plot_accuracy_changes(history)
		cp.plot_per()

	def train_keras_model_with_embedding(self, number_of_epochs):
		print(self.train_data_X.shape)
		model = Sequential([
			Embedding(self.number_of_context_words*self.vocab_size, self.number_of_context_words*self.projection_size),
			Dense(self.number_of_context_words*self.projection_size),
			Activation('linear'),
			Dense(self.number_of_hidden_layer_neurons),
			Activation('sigmoid'),
			Dense(self.vocab_size),
			Activation('softmax')
		])
		print(model.summary())
		model.compile(optimizer='sgd',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

		cp = calculate_perplexity(self.train_data_X, self.train_data_Y)
		history = model.fit(self.train_data_X, self.train_data_Y, epochs=number_of_epochs, batch_size=256, validation_data=(self.test_data_X, self.test_data_Y), callbacks=[cp])
		self.plot_accuracy_changes(history)
		cp.plot_per()

	def train(self, number_of_epochs):
		print( self.number_of_hidden_layer_neurons, self.number_of_context_words, self.learning_rate)
		if self.use_glove == None:
			self.train_keras_model_with_embedding(number_of_epochs)
		else:
			self.train_keras_model(number_of_epochs)