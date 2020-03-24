import os 
import nltk 

def download_the_dataset():
    nltk.download('movie_reviews')

def print_class_counts(path_to_downloaded_data):
    print('Number of Neg. Examples', len(os.listdir(path_to_downloaded_data+'neg/')))
    print('Number of Pos. Examples', len(os.listdir(path_to_downloaded_data+'pos/')))

def print_basic_statistics(dataset):
    pos_lens = []
    neg_lens = []

    neg_no_counts = []
    pos_no_counts = []

    neg_pronoun_counts = []
    pos_pronoun_counts = []

    for ind in range(len(dataset['X'])):
        lowered_text = dataset['X'][ind].lower()
        no_counts = lowered_text.count('no')
        pronount_counts = lowered_text.count(' you ')+lowered_text.count(' me ')+lowered_text.count(' i ')+lowered_text.count(" you're ")+lowered_text.count(" i'm ")+lowered_text.count(" i've ")+lowered_text.count(" you've ")
        if dataset['Y'][ind] == 'neg':
            neg_lens.append(len(dataset['X'][ind]))
            neg_no_counts.append(no_counts)
            neg_pronoun_counts.append(pronount_counts)
        elif dataset['Y'][ind] == 'pos':
            pos_lens.append(len(dataset['X'][ind]))
            pos_no_counts.append(no_counts)
            pos_pronoun_counts.append(pronount_counts)
    
    print('Avg. pos character lengths:', sum(pos_lens)/len(pos_lens))
    print('Avg. neg character lengths:', sum(neg_lens)/len(neg_lens))

    print('Avg. pos no counts:', sum(pos_no_counts)/len(pos_no_counts))
    print('Avg. neg no counts:', sum(neg_no_counts)/len(neg_no_counts))

    print('Avg. pos pronoun counts:', sum(pos_pronoun_counts)/len(pos_pronoun_counts))
    print('Avg. neg pronoun counts:', sum(neg_pronoun_counts)/len(neg_pronoun_counts))



def load_the_dataset(path_to_downloaded_data):
    dataset = {'X': [], 'Y': []}
    
    neg_file_paths = [path_to_downloaded_data+'neg/'+val for val in os.listdir(path_to_downloaded_data+'neg/')]
    pos_file_paths = [path_to_downloaded_data+'pos/'+val for val in os.listdir(path_to_downloaded_data+'pos/')]

    for file_name in neg_file_paths:
        with open(file_name, 'r') as fp:
            content = fp.read()
            dataset['X'].append(content)
            dataset['Y'].append('neg')

    for file_name in pos_file_paths:
        with open(file_name, 'r') as fp:
            content = fp.read()
            dataset['X'].append(content)
            dataset['Y'].append('pos')
    
    return dataset

def get_dataset():
    download_the_dataset()
    path_to_downloaded_data = '/Users/User/nltk_data/corpora/movie_reviews/'
    print_class_counts(path_to_downloaded_data)
    dataset = load_the_dataset(path_to_downloaded_data)
    print_basic_statistics(dataset)
    return dataset