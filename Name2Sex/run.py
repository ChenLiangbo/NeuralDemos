#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import data_util
import datetime

# http://blog.topspeedsnail.com/archives/10833
name_dataset ='./dataset/name.csv'

train_x,train_y = data_util.read_data(name_dataset)
max_name_length  = max([len(name) for name in train_x])
max_name_length = 8
vocabulary = data_util.get_vocabulary(train_x)
train_x_vec,vocab,vocabulary_list = data_util.vectorize(train_x,vocabulary,max_name_length)
print("--------------------read data okay----------------------------------")


input_size = max_name_length
num_classes = 2

batch_size = 64
num_batch = len(train_x_vec) // batch_size

X = tf.placeholder(tf.int32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, num_classes])

dropout_keep_prob = tf.placeholder(tf.float32)

def neural_network(vocabulary_size, embedding_size=128, num_filters=128):
	# embedding layer
	with tf.device('/cpu:0'), tf.name_scope("embedding"):
		W = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
		embedded_chars = tf.nn.embedding_lookup(W, X)
		embedded_chars_expanded = tf.expand_dims(embedded_chars, -1) #[?,8,128,1]
	# print("embedded_chars_expanded = ",embedded_chars_expanded.get_shape())
	# convolution + maxpool layer
	filter_sizes = [3,4,5]
	pooled_outputs = []
	for i, filter_size in enumerate(filter_sizes):
		with tf.name_scope("conv-maxpool-%s" % filter_size):
			filter_shape = [filter_size, embedding_size, 1, num_filters]
			# print("filter_shape = ",filter_shape)
			W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
			b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
			conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
			# print("conv = ",conv.get_shape())
			h = tf.nn.relu(tf.nn.bias_add(conv, b))
			pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
			# print("pooled = ",pooled.get_shape())
			pooled_outputs.append(pooled)

	num_filters_total = num_filters * len(filter_sizes)
	h_pool = tf.concat(3, pooled_outputs)
	h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
	# print("h_pool_flat = ",h_pool_flat.get_shape())  [?,304]
	# dropout
	with tf.name_scope("dropout"):
		h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
	# output
	with tf.name_scope("output"):
		W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
		b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
		output = tf.nn.xw_plus_b(h_drop, W, b) # [?,2]
		# print("output = " ,output.get_shape())
		
	return output
# 训练
def train_neural_network(epoch):
	output = neural_network(len(vocabulary_list))

	optimizer = tf.train.AdamOptimizer(1e-3)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
	grads_and_vars = optimizer.compute_gradients(loss)
	train_op = optimizer.apply_gradients(grads_and_vars)

	saver = tf.train.Saver(tf.global_variables())
	# saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		try:
			saver.restore(sess,"./model/name2sex.data")
			print("load model okay!")
		except Exception as ex:
			print("[Exception Information] ",str(ex))

		for e in range(epoch):
			for i in range(num_batch):
				batch_x = train_x_vec[i*batch_size : (i+1)*batch_size]
				batch_y = train_y[i*batch_size : (i+1)*batch_size]
				_, loss_ = sess.run([train_op, loss], feed_dict={X:batch_x, Y:batch_y, dropout_keep_prob:0.5})
				
				if i % 1000 == 0:
					now = datetime.datetime.now().strftime('%d %H:%M:%S')
					print("epoch = %d, step = %d, loss = %f ,time = %s" % (e,i,loss_,now))
				#break
			# break
			# 保存模型

			saver.save(sess, "./model/name2sex.data")

train_neural_network(epoch = 1000)
print('--------------------- train model okay ---------------------------')
# 使用训练的模型
def detect_sex(name_list):
	x = []
	for name in name_list:
		name_vec = []
		for word in name:
			name_vec.append(vocab.get(word))
		while len(name_vec) < max_name_length:
			name_vec.append(0)
		x.append(name_vec)

	output = neural_network(len(vocabulary_list))

	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		# 恢复前一次训练
		saver.restore(sess,'./model/name2sex.data')


		predictions = tf.argmax(output, 1)
		res = sess.run(predictions, {X:x, dropout_keep_prob:1.0})

		i = 0
		for name in name_list:
			print(name, '女' if res[i] == 0 else '男')
			i += 1

detect_sex(["白富美", "高帅富", "王婷婷", "田野"])
