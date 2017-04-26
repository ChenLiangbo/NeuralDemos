#!usr/bin/env/python 
# -*- coding: utf-8 -*-

# dataset https://pan.baidu.com/s/1o7QlUhO   poetry.txt
# http://blog.topspeedsnail.com/archives/10542
# sudo python3 -m pip install tensorflow==0.12.0
# http://blog.csdn.net/u014365862/article/details/53868544
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
# 按诗的字数排序
poetrys = sorted(poetrys,key=lambda line: len(line))
print('唐诗总数: ', len(poetrys))

# 统计每个字出现次数
all_words = []
for poetry in poetrys:
	all_words += [word for word in poetry]
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)
 
# 取前多少个常用字
words = words[:len(words)] + (' ',)

# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words))))
# 把诗转换为向量形式，参考TensorFlow练习1
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]
#[[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
#[339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
#....]

# 每次取64首诗进行训练
batch_size = 4
n_chunk = len(poetrys_vector) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
	start_index = i * batch_size
	end_index = start_index + batch_size
 
	batches = poetrys_vector[start_index:end_index]
	length = max(map(len,batches))
	xdata = np.full((batch_size,length), word_num_map[' '], np.int32)
	for row in range(batch_size):
		xdata[row,:len(batches[row])] = batches[row]
	ydata = np.copy(xdata)
	ydata[:,:-1] = xdata[:,1:]
	"""
	xdata             ydata
	[6,2,4,6,9]       [2,4,6,9,9]
	[1,4,2,8,5]       [4,2,8,5,5]
	"""
	x_batches.append(xdata)
	y_batches.append(ydata)
	if i > 2000:
		break
print("xdata = ",xdata,type(xdata))
print("-"*80)
print("ydata = ",ydata,type(ydata))
print('='*80)

def to_poem(ddata,words):
	shape = ddata.shape
	# print("shape = ",shape)
	poem = []
	for i in range(shape[0]):
		line = []
		for j in range(shape[1]):
			line.append(words[ddata[i,j]])
		poem.append(line)
	return poem

def print_poem(poem):
	print("-"*30 + 'poem' + '-'*30)
	for line in poem:
		print(line)
	print("-"*30 + 'poem' + '-'*30)

x = to_poem(xdata,words)
# print("x = ",x)
y = to_poem(ydata,words)
print_poem(x)
print_poem(y)