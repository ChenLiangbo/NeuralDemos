#!usr/bin/env/python 
# -*- coding: utf-8 -*-

# dataset https://pan.baidu.com/s/1o7QlUhO   poetry.txt
# http://blog.topspeedsnail.com/archives/10542

import collections
import numpy as np
import tensorflow as tf
import pickle

# --------------------------load dataset---------------------------------------
dataset = './dataset/'
fpx = open(dataset + 'x_batches.pkl','rb')
x_batches = pickle.load(fpx)
fpx.close()

fpy = open(dataset + 'y_batches.pkl','rb')
y_batches = pickle.load(fpy)
fpy.close()

#---------------------------------------RNN--------------------------------------#
batch_size = 64
input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])
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
		# tf.Variable
		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [len(words)+1, rnn_size])
			inputs = tf.nn.embedding_lookup(embedding, input_data)
 
	outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
	output = tf.reshape(outputs,[-1, rnn_size])
 
	logits = tf.matmul(output, softmax_w) + softmax_b
	probs = tf.nn.softmax(logits)
	return logits, last_state, probs, cell, initial_state
#训练

def train_neural_network():
	logits, last_state, _, _, _ = neural_network()
	targets = tf.reshape(output_targets, [-1])
	loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], len(words))
	cost = tf.reduce_mean(loss)
	learning_rate = tf.Variable(0.0, trainable=False)
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.apply_gradients(zip(grads, tvars))
 
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
 
		# saver = tf.train.Saver(tf.all_variables())
		saver = tf.train.Saver(tf.global_variables())
 
		for epoch in range(50):
			sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
			n = 0
			for batche in range(n_chunk):
				train_loss, _ , _ = sess.run([cost, last_state, train_op], feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})
				n += 1
				if batche % 30 == 0:
				    print("epoch =%d, batche = %d, train_loss = %f" % (epoch, batche, train_loss))
			if epoch % 10 == 0:
				saver.save(sess, './model/poetry.module', global_step=epoch)
 
train_neural_network()