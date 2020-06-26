from neural_network import NeuralNetwork 
from constants import glove_embedding_file_name
from data_reader import glove_embeddings_reader

def run(train_data, test_data):    
    vocab_size = len(set(train_data)) + 2
    projection_size = 50

    word_to_index = {}
    word_ind = 0
    for this_word in set(train_data):
        word_to_index[this_word] = word_ind
        word_ind += 1
    word_to_index['<S>'] = word_ind
    word_to_index['<UNK>'] = word_ind+1
    
    ncw = 4
    nhln = 35
    lr = 0.2
    print('Learning Rate = {}, Number of context words = {}, Number of hidden layer nodes = {}'.format(lr, ncw, nhln))
    nn = NeuralNetwork(number_of_hidden_layer_neurons=nhln, 
                    number_of_context_words=ncw,
                    learning_rate=lr,  
                    use_glove=None, 
                    vocab_size=vocab_size, 
                    projection_size=projection_size, 
                    train_data=train_data, 
                    test_data=test_data, 
                    word_to_index=word_to_index)
    nn.prep_data_for_test_and_train()
    nn.train(number_of_epochs = 20)