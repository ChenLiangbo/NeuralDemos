#!usr/bin/env/python 
# -*- coding: utf-8 -*-


def read_data(name_dataset):
    train_x = []
    train_y = []
    with open(name_dataset,'r',encoding = 'utf-8') as f:
        first_line = True
        for line in f:
            if first_line:
            	first_line = False
            	continue
            sample = line.strip().split(',') # sample[0] 姓名,sample[1] 性别
            if len(sample) == 2:
            	train_x.append(sample[0])
            	if sample[1] == '男':
            	    train_y.append([0,1])  # 男生 [0,1]  0
            	else:
            		train_y.append([1,0])  # 女生 [1,0]  1
    return train_x,train_y

def get_vocabulary(train_x):
    counter = 0
    vocabulary = {} # 统计所有出现的汉字 以及汉字的频率 vocabulary["婷"] = 6
    for name in train_x:
    	counter = counter + 1
    	tokens  = [word for word in name]
    	for word in tokens:
    	    if word in vocabulary:
    	        vocabulary[word] += 1
    	    else:
    	        vocabulary[word] = 1
    return vocabulary

# 字符串转为向量形式
# In:train_x :list,vocabulary: dict
def vectorize(train_x,vocabulary,max_name_length):
    vocabulary_list = [' '] + sorted(vocabulary,key = vocabulary.get,reverse = True)
    vocab = dict([(x,y) for (y,x) in enumerate(vocabulary_list)]) # vocab[6] = '婷'
    train_x_vec = []
    for name in train_x:
        name_vect = []
        for word in name:
        	name_vect.append(vocab.get(word))    # 向量化的方式是 每个汉字出现的次数
        while(len(name_vect) < max_name_length):
            name_vect.append(0)
        train_x_vec.append(name_vect)
    # print("train_x_vec = ",len(train_x_vec),len(train_x_vec[10]))
    return train_x_vec,vocab,vocabulary_list

if __name__ == '__main__':

    name_dataset ='./dataset/name.csv'
    
    train_x,train_y = read_data(name_dataset)
    print("train_x = ",train_x[10],len(train_x)) # train_x 姓名 [str1,str2,..]
    print("train_y = ",train_y[10],len(train_y)) # train_y 性别 [[0,1],[1,0],,...]
    
    max_name_length  = max([len(name) for name in train_x])
    print("max_name_length = ",max_name_length)
    max_name_length  = 8  #此处似乎没有必要
    # 可以打乱数据集
    vocabulary = get_vocabulary(train_x)
    
    # dataset size 
    # train_x_vec = (None,max_name_length)  max_name_length = 8 ？
    # train_y     = (None,2)
    train_x_vec,vocab,vocabulary_list = vectorize(train_x,vocabulary,max_name_length)
    print("train_x_vec = ",len(train_x_vec),len(train_x_vec[0]))
    print("train_x_vec = ",train_x_vec[0:2])
    print("train_y = ",train_y[0:2])
