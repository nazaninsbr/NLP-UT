from constants import train_file_name, test_file_name, glove_embedding_file_name
from data_reader import read_file
from pre_processing import preform_pre_processing

from Q1 import run as run_q1
from Q2 import run as run_q2 

def main():
    train_data, test_data = read_file(train_file_name), read_file(test_file_name)
    cleaned_train_data, cleaned_test_data = preform_pre_processing(train_data), preform_pre_processing(test_data)
    
    print('========== Question 1 ==========')
    run_q1(cleaned_train_data, cleaned_test_data)
    print('========== Question 2 ==========')
    run_q2(cleaned_train_data, cleaned_test_data)

main()