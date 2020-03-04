import os

def read_file_content(file_path):
    with open(file_path, 'r') as fp:
        content = fp.read()
    return content

def read_the_data(data_type):
    return_data = {}
    if data_type == 'train':
        path = '../CA1_Data/train/'
        folders_to_read = os.listdir(path)
        for folder_name in folders_to_read:
            return_data[folder_name] = {}
            file_names = os.listdir(path+folder_name+'/')
            for f_name in file_names:
                file_path = path+folder_name+'/'+f_name
                return_data[folder_name][f_name] = read_file_content(file_path)
    elif data_type == 'test':
        path = '../CA1_Data/test/'
        file_names = os.listdir(path)
        return_data['UnknownLabel'] = {}
        for f_name in file_names:
            file_path = path+f_name
            return_data['UnknownLabel'][f_name] = read_file_content(file_path) 
    return return_data

def first_look_at_the_data():
    train_data_dict = read_the_data('train')
    test_data_dict = read_the_data('test')

    number_of_train_instances = sum([len(train_data_dict[k].keys()) for k in train_data_dict.keys()])
    number_of_test_instances = len(test_data_dict['UnknownLabel'].keys())
    print('Number of training data:', number_of_train_instances)
    print('Number of test data:', number_of_test_instances)
    average_char_length_of_train = 0
    average_char_length_of_test = 0
    for k in test_data_dict['UnknownLabel'].keys():
        average_char_length_of_test += len(test_data_dict['UnknownLabel'][k])
    average_char_length_of_test = average_char_length_of_test/number_of_test_instances

    for k in train_data_dict.keys():
        for f_name in train_data_dict[k].keys():
            average_char_length_of_train += len(train_data_dict[k][f_name])
    average_char_length_of_train = average_char_length_of_train/number_of_train_instances

    print('Avg. Character Length of Train instances:', average_char_length_of_train)
    print('Avg. Character Length of Test instances:', average_char_length_of_test)
    