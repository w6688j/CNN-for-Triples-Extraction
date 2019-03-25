from model.networks import *

Model = CNN([2, 2, 5])
Model.train(data_path='data/assignment_training_data_word_segment.json', maxepoch=250, continue_train=False,
            trained_steps=0)
