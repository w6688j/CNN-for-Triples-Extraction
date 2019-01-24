from model.networks import *
Model = CNN([2, 2, 5])
Model.test(data_path='.\\data\\assignment_test_data_word_segment.json')
