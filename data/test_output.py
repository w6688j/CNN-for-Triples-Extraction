import json
import os

#data_path = os.getcwd() + '/assignment_training_data_word_segment.json'
#data = json.load(open(data_path,'r'))
#print(type(data))
#for i in data:
"""  
    i.pop('sentence')
    i.pop('indexes')
    i.pop('times')
    i.pop('attributes')
    i.pop('values')
    i.pop('words')
    test = []
    i['results'] = test.append([1,2,3])
    #print(test)
    i['results'] = test.append([4,5,6])
    #print(test)
    i['results'] = test
"""
    #print(i)
#print(data)
#print(json.dumps(data))
"""
with open('data.json','w') as json_file:
    json_file.write(json.dumps(data))

data_more = json.load(open(os.getcwd()+'/data.json','r'))
print(type(data))
for i in data:
    print(i)
"""


class generate_my_json():
    def __init__(self,data_path_base=os.getcwd(),filename='\\data\\assignment_training_data_word_segment.json',Train_or_test = True):
        self.data = json.load(open(data_path_base+filename,'r'))
        self.partial_new_data = []
        self.Train_or_test = Train_or_test
        self.i = 0
        self.flag = 0
        
    def input_the_data(self, new_data, Finish_partial=False, Finish_all=False):
        if Finish_all:
            with open('result.json', 'w') as json_file:
                json_file.write(json.dumps(self.data))
        else:
            if self.Train_or_test:
                if self.flag == 0:
                    self.data[self.i].pop('sentence')
                    self.data[self.i].pop('indexes')
                    self.data[self.i].pop('times')
                    self.data[self.i].pop('attributes')
                    self.data[self.i].pop('values')
                    self.data[self.i].pop('words')           
                    self.data[self.i].pop('results')
                    self.flag = 1
                    self.partial_new_data.append(new_data)
                else:
                    self.partial_new_data.append(new_data)
                    if Finish_partial:
                        self.data[self.i]['results'] = self.partial_new_data
                        self.partial_new_data = []
                        self.flag = 0
                        self.i = self.i + 1
            else:
                if self.flag == 0:
                    self.data[self.i].pop('indexes')
                    self.data[self.i].pop('times')
                    self.data[self.i].pop('attributes')
                    self.data[self.i].pop('values')         
                    self.data[self.i].pop('results')
                    self.flag = 1
                    self.partial_new_data.append(new_data)
                else:
                    self.partial_new_data.append(new_data)
                    if Finish_partial:
                        i['results'] = self.partial_new_data
                        self.partial_new_data = []
                        self.flag = 0
                        self.i = self.i + 1
