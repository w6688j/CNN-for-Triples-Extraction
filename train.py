from model.networks import *
Model = CNN([2, 1, 5])
Model.train(data_path='.\\data\\assignment_training_data_word_segment.json', maxepoch=200)
