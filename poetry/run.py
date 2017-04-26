#!usr/bin/env/python 
# -*- coding: utf-8 -*-

# dataset https://pan.baidu.com/s/1o7QlUhO   #全唐诗(43030首)
# http://blog.topspeedsnail.com/archives/10542

import collections
import numpy as np
import tensorflow as tf
import time


print("-------------------------------数据预处理---------------------------")
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
word_num_map = dict(zip(words, range(len(words))))  # {"人":5,""}
# 把诗转换为向量形式，参考TensorFlow练习1
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]
#[[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
#[339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
#....]
 

batch_size = 1
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
 
print("words = ",len(words))
# import pickle
# fp = open("./caogao/words.pkl",'wb')
# pickle.dump(words,fp,True)
# fp.close()
#---------------------------------------RNN--------------------------------------#
 
input_data = tf.placeholder(tf.int32, [batch_size, None])    # [64,None]
output_targets = tf.placeholder(tf.int32, [batch_size, None])# [64,None]
# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
	if model == 'rnn':
		cell_fun = tf.nn.rnn_cell.BasicRNNCell
	elif model == 'gru':
		cell_fun = tf.nn.rnn_cell.GRUCell
	elif model == 'lstm':
		cell_fun = tf.nn.rnn_cell.BasicLSTMCell
 
	cell = cell_fun(rnn_size, state_is_tuple=True)
	cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
 
	initial_state = cell.zero_state(batch_size, tf.float32)
 
	with tf.variable_scope('rnnlm'):
		softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)+1])
		softmax_b = tf.get_variable("softmax_b", [len(words)+1])
		with tf.device("/cpu:0"):
			# embedding = (6111,128) inputs = (1,?,128)
			embedding = tf.get_variable("embedding", [len(words)+1, rnn_size])
			inputs = tf.nn.embedding_lookup(embedding, input_data)
			# print("inputs.get_shape = ",inputs.get_shape())
 
	outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
	# print("outputs.get_shape = ",outputs.get_shape()) # (1,?,128)
	output = tf.reshape(outputs,[-1, rnn_size])       # (?,128)
	# print("output.get_shape = ",output.get_shape())

 
	logits = tf.matmul(output, softmax_w) + softmax_b  # (?,6111)
	# print("logits.get_shape = ",logits.get_shape())
	probs = tf.nn.softmax(logits)
	# print("probs.get_shape = ",probs.get_shape())       #(?,6111)
	return logits, last_state, probs, cell, initial_state

print("#-------------------------------生成古诗---------------------------------#")
# array([[1, 2, 3, 4],
#        [3, 4, 5, 6]])                                  
# array([ 1,  3,  6, 10, 13, 17, 22, 28])  np.cumsum(a)
# (2,4) -> (8,)
def get_word(weights):
	index = weights.argmax(axis = 1)
	return words[int(index)]

# 使用训练完成的模型
def gen_poetry():
	def to_word(weights):
		t = np.cumsum(weights)
		# print("t = ",t.shape)  # (6111,)
		s = np.sum(weights)
		# print("s = ",s)  # s = 1.0
		sample = int(np.searchsorted(t, np.random.rand(1)*s))
		# print("sample = ",sample)  # 随机数 从t中随机抽取
		# print("words = ",len(words))  # 6110
		return words[sample]   # 从诗歌汉子数据集中选取一个返回
 
	_, last_state, probs, cell, initial_state = neural_network()
 
	with tf.Session() as sess:

		init = tf.global_variables_initializer()
		# init = tf.initialize_all_variables()
		sess.run(init)
 
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(sess,'./model/poetry.data')
		print("restore model successfully !")
		state_ = sess.run(cell.zero_state(1, tf.float32))
		x = np.array([list(map(word_num_map.get, '['))])  # x.shape = (1,1)
		# print("x = ",x) # x = [[2]]
		[probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
		# print("probs_ = ",probs_.shape)  # (1,6111)
		# fp = open('./caogao/probs.pkl','wb')
		# pickle.dump(probs_,fp,True)
		# fp.close()

		word = to_word(probs_)
		# word = get_word(probs_)
		print("first word = ",word,words[-1])
		#word = words[np.argmax(probs_)]
		poem = ''
		i = 0
		while word != ']':
			if word != '[':
				poem += word
			x = np.zeros((1,1))
			x[0,0] = word_num_map[word]
			[probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
			# word = to_word(probs_)
			word = get_word(probs_)
			# print("word = ",word," poem = ",poem)
			#word = words[np.argmax(probs_)]
			# break
			i = i + 1
			if i > 200:
				break
		return poem
poem = gen_poetry()
poem = poem.split('。')
print("-"*50)
for p in poem:
	print(p)
print('-'*50)


print("--------------------------------生成藏头诗--------------------------------#")
def gen_poetry_with_head(head):
	def to_word(weights):
		t = np.cumsum(weights)
		s = np.sum(weights)
		sample = int(np.searchsorted(t, np.random.rand(1)*s))
		return words[sample]
 
	_, last_state, probs, cell, initial_state = neural_network()
 
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
 
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(sess,'./model/poetry.data')	
 
		state_ = sess.run(cell.zero_state(1, tf.float32))
		poem = ''
		i = 0
		for word in head:
			while word != '，' and word != '。':
				poem += word
				x = np.array([list(map(word_num_map.get, word))])
				[probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
				# word = to_word(probs_)
				word = get_word(probs_)
				time.sleep(1)
			if i % 2 == 0:
				poem += '，'
			else:
				poem += '。'
			i += 1
		return poem
 
# print(gen_poetry_with_head('一二三四'))