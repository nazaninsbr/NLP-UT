def run_data_section_codes():
    from data_readers import first_look_at_the_data
    first_look_at_the_data()

def run_question_1_code():
    from q1 import run_q1_code
    from data_readers import read_the_data

    train_data_dict = read_the_data('train')
    run_q1_code(train_data_dict)

def run_question_2_code():
    from q1 import clean_text
    from q2 import run_q2_code
    from data_readers import read_the_data

    train_data_dict = read_the_data('train')
    train_cleaned_data = clean_text(train_data_dict, do_lemmatization=False)
    run_q2_code(train_cleaned_data)

def run_question_3_code():
    from q1 import clean_text
    from q2 import train_lang_models_and_prep_test_data
    from data_readers import read_the_data
    from q3 import run_q3_code

    train_data_dict = read_the_data('train')
    train_cleaned_data = clean_text(train_data_dict, do_lemmatization=True)
    all_models, preped_test_X_Y = train_lang_models_and_prep_test_data(train_cleaned_data)
    run_q3_code(all_models, preped_test_X_Y)

def run_question_4_code():
    from q1 import clean_text
    from q2 import train_lang_models_and_prep_test_data
    from data_readers import read_the_data
    from q4 import run_q4_code

    train_data_dict = read_the_data('train')
    test_data_dict = read_the_data('test')
    train_cleaned_data = clean_text(train_data_dict, do_lemmatization=True)
    test_data_cleaned = clean_text(test_data_dict, do_lemmatization=True)
    all_models, _ = train_lang_models_and_prep_test_data(train_cleaned_data, no_label_test_dict = True)
    best_model = all_models['Word']['Unigram']
    run_q4_code(best_model, test_data_cleaned)

def main():
    print('##################### Data #####################')
    run_data_section_codes()
    print('##################### Question 1 #####################')
    run_question_1_code()
    print('##################### Question 2 #####################')
    run_question_2_code()
    print('##################### Question 3 #####################')
    run_question_3_code()
    print('##################### Question 4 #####################')
    run_question_4_code()

if __name__ == '__main__':
    main()