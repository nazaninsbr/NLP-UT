from neural_network import NeuralNetwork 
from constants import glove_embedding_file_name
from data_reader import glove_embeddings_reader

def run(train_data, test_data):
	glove_embedding = glove_embeddings_reader(glove_embedding_file_name)
	
	vocab_size = len(set(train_data)) + 2
	projection_size = 50

	word_to_index = {}
	word_ind = 0
	for this_word in set(train_data):
		word_to_index[this_word] = word_ind
		word_ind += 1
	word_to_index['<S>'] = word_ind
	word_to_index['<UNK>'] = word_ind+1
	
	nn = NeuralNetwork(number_of_hidden_layer_neurons=35, 
						number_of_context_words=4,
						learning_rate=0,  
						use_glove=glove_embedding, 
						vocab_size=vocab_size, 
						projection_size=projection_size, 
						train_data=train_data, 
						test_data=test_data, 
						word_to_index=word_to_index)
	nn.prep_data_for_test_and_train()
	nn.set_parameters(number_of_hidden_layer_neurons=35, number_of_context_words=4, learning_rate=0.02)
	nn.train(number_of_epochs = 20)

	for lr in [0.01, 0.03, 0.1]:
		ncw = 4
		nhln = 35
		print('Learning Rate = {}, Number of context words = {}, Number of hidden layer nodes = {}'.format(lr, ncw, nhln))
		nn.set_parameters(number_of_hidden_layer_neurons=nhln, number_of_context_words=ncw, learning_rate=lr)
		nn.train(number_of_epochs = 20)

	for ncw in [4, 3, 2]:
		lr = 0.01
		nhln = 35
		print('Learning Rate = {}, Number of context words = {}, Number of hidden layer nodes = {}'.format(lr, ncw, nhln))
		nn = NeuralNetwork(number_of_hidden_layer_neurons=nhln, 
						number_of_context_words=ncw,
						learning_rate=lr,  
						use_glove=glove_embedding, 
						vocab_size=vocab_size, 
						projection_size=projection_size, 
						train_data=train_data, 
						test_data=test_data, 
						word_to_index=word_to_index)
		nn.prep_data_for_test_and_train()
		nn.train(number_of_epochs = 20)
	

	nn = NeuralNetwork(number_of_hidden_layer_neurons=35, 
						number_of_context_words=4,
						learning_rate=0,  
						use_glove=glove_embedding, 
						vocab_size=vocab_size, 
						projection_size=projection_size, 
						train_data=train_data, 
						test_data=test_data, 
						word_to_index=word_to_index)
	nn.prep_data_for_test_and_train()
	for nhln in [50, 100, 150]:
		lr = 0.01
		ncw = 4
		print('Learning Rate = {}, Number of context words = {}, Number of hidden layer nodes = {}'.format(lr, ncw, nhln))
		nn.set_parameters(number_of_hidden_layer_neurons=nhln, number_of_context_words=ncw, learning_rate=lr)
		nn.train(number_of_epochs = 20)
		