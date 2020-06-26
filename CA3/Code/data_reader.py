import numpy as np 

def read_file(file_name):
    with open(file_name, 'r') as fp:
        data = fp.read()
    return data

def glove_embeddings_reader(file_path):
    emd = {}
    with open(file_path, 'r') as fp:
        data = fp.read().split('\n')
    for line in data:
        line_pieces = line.split(' ')
        if len(line_pieces)<1:
            pass
        word = line_pieces[0]
        this_words_embd = np.array([float(val) for val in line_pieces[1:]])
        if this_words_embd.shape[0] == 0:
            continue 
        emd[word] = this_words_embd
    
    emd['<S>'] = np.mean([emd[k] for k in emd.keys()], axis=0)
    emd['<UNK>'] = np.mean([emd[k] for k in emd.keys()], axis=0)
    return emd