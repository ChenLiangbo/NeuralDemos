#!usr/bin/env/python 
# -*- coding: utf-8 -*-

# dataset https://pan.baidu.com/s/1o7QlUhO   poetry.txt
# http://blog.topspeedsnail.com/archives/10542

import collections
import numpy as np
import tensorflow as tf
 
#-------------------------------数据预处理---------------------------#
 
dataset = './dataset/'
poetry_file = dataset + 'poetry.txt'
 
# 诗集 从文件中生成古诗词列表
def read_data(poetry_file):
    poetrys = []
    with open(poetry_file, "r", encoding='utf-8',) as f:
    	for line in f:
    		try:
    			title, content = line.strip().split(':')
    			content = content.replace(' ','')
    			if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
    				continue
    			if len(content) < 5 or len(content) > 79:
    				continue
	    		content = '[' + content + ']'
	    		# print("content = ",content,type(content))  # string
	    		# [寒随穷律变，春逐鸟声开。初风飘带柳，晚雪间花梅。碧林青旧竹，绿沼翠新苔。芝田初雁去，绮树巧莺来。]
	    		# break
	    		poetrys.append(content)
	    	except Exception as e:
	    		pass
    return poetrys
poetrys = read_data(poetry_file)
print("poetrys = ",poetrys[0:3])


# 按诗的字数排序
poetrys = sorted(poetrys,key=lambda line: len(line))
print('唐诗总数: ', len(poetrys))
    
# 统计每个字出现次数
all_words = []
for poetry in poetrys:
    # poetry 表示一首诗  poetrys 所有诗的集合  all_words 诗词中出现的汉子去重复集合
    all_words += [word for word in poetry]
# print("all_words = ",len(all_words))
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)
# print("words = ",len(words))
# 取前多少个常用字
words = words[:len(words)] + (' ',)
# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words))))    # {"人":5,""}
# 把诗转换为向量形式 一首诗转换为一个向量 因此这些向量长度不一样
# 将常用的汉字编序号，越靠前越常用，使用序号替换汉字生成向量 包括 ，。 [ ] 字符
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]
print("poetrys = ",poetrys[0:3])
print("poetrys_vector = ",poetrys_vector[0:3])
# print(poetrys_vector[99:102])
# print(poetrys[0:2])
print("-"*80)
#[[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
#[339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
#....]

# 每次取64首诗进行训练 生成的数据 行数固定 列数不固定
batch_size = 64
n_chunk = len(poetrys_vector) // batch_size
print("n_chunk = ",n_chunk)
x_batches = []
y_batches = []
for i in range(2,n_chunk):
	start_index = i * batch_size
	end_index = start_index + batch_size
	batches = poetrys_vector[start_index:end_index]
	length = max(map(len,batches))    #选取该batch里最长的作为该batch的猎术
	# print("length = ",length,", word_num_map[' '] = ",word_num_map[' '])
	#  word_num_map[' '] = 6109 每一首诗长度不够的使用 6019从后面补全
	xdata = np.full((batch_size,length), word_num_map[' '], np.int32)  # shape = (64,length)
	# print("xdata = ",xdata.shape) # shape = (batch_size,length)
	# 模型生成的诗歌没有意义，本质上是因为数据集制作的时候给定的输出没有意义的
	for row in range(batch_size):
		xdata[row,:len(batches[row])] = batches[row]
	ydata = np.copy(xdata)
	ydata[:,:-1] = xdata[:,1:] # 倒数第二列开始每列向前平移一列  最后一列不变
	x_batches.append(xdata)
	y_batches.append(ydata)
 
	# print(xdata)
	# print("-"*80)
	# print(ydata)
	# break
import pickle
# dump(object, file) 
# load(file)
# 文件可以是实际的物理文件，
# 但也可以是任何类似于文件的对象，
# 这个对象具有 write() 方法

fpx = open(dataset + 'x_batches.pkl','wb')
pickle.dump(x_batches,fpx)
fpx.close()

fpy = open(dataset + 'y_batches.pkl','wb')
pickle.dump(y_batches,fpy)
fpy.close()


'''
xdata
[[   3 1144  947 1322  522  661    1    2]
 [   3 1364  104   18   83  583    1    2]
 [   3  649   48    9 2143  392    1    2]
 [   3  168 2324 1629 1077 2324    1    2]]
ydata
[[1144  947 1322  522  661    1    2    2]
 [1364  104   18   83  583    1    2    2]
 [ 649   48    9 2143  392    1    2    2]
 [ 168 2324 1629 1077 2324    1    2    2]]

RNN model 
input : [3     1144 947   1322  522    661    1    2]
output: [1144  947  1322  522   661    1      2    2]

'''