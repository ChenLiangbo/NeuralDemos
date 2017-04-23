#!usr/bin/env/python 
# -*- coding: utf-8 -*-

# http://blog.csdn.net/u014365862/article/details/53869732
name_dataset = './dataset/name.csv'  
   
train_x = []  
train_y = []  
with open(name_dataset, 'r') as f:  
    first_line = True  
    for line in f:  
        if first_line is True:  
            first_line = False  
            continue  
        sample = line.strip().split(',')  
        if len(sample) == 2:  
            train_x.append(sample[0])  
            if sample[1] == '男':  
                train_y.append([0, 1])  # 男  
            else:  
                train_y.append([1, 0])  # 女  
   
max_name_length = max([len(name) for name in train_x])  
print("最长名字的字符数: ", max_name_length)  
max_name_length = 8  
   
# 数据已shuffle  
#shuffle_indices = np.random.permutation(np.arange(len(train_y)))  
#train_x = train_x[shuffle_indices]  
#train_y = train_y[shuffle_indices]  
   
# 词汇表（参看聊天机器人练习）  
counter = 0  
vocabulary = {}  
for name in train_x:  
    counter += 1  
    tokens = [word for word in name]  
    for word in tokens:  
        if word in vocabulary:  
            vocabulary[word] += 1  
        else:  
            vocabulary[word] = 1  
   
vocabulary_list = [' '] + sorted(vocabulary, key=vocabulary.get, reverse=True)  
print(len(vocabulary_list))  
   
# 字符串转为向量形式  
vocab = dict([(x, y) for (y, x) in enumerate(vocabulary_list)])  
train_x_vec = []  
for name in train_x:  
    name_vec = []  
    for word in name:  
        name_vec.append(vocab.get(word))  
    while len(name_vec) < max_name_length:  
        name_vec.append(0)  
    train_x_vec.append(name_vec)  
   