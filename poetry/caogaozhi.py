#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import pickle

fp = open("./caogao/words.pkl",'rb')
words = pickle.load(fp)
fp.close()
print("words = ",len(words))


fp = open('./caogao/probs.pkl','rb')
probs = pickle.load(fp)
fp.close()
# print("probs = ",probs.shape) (1, 6111)
pmax = np.max(probs)
print("pmax = ",pmax)
i = probs.argmax(axis = 1)
print(" i = ",i,probs[0,i])
# from matplotlib import pyplot as plt
# plt.plot(probs[0,:],'r-')
# plt.show()

def get_word(weights,words):
	index = weights.argmax(axis = 1)
	return words[int(index)]

def to_word(weights,words):
	t = np.cumsum(weights)
	print("t = ",t.shape)  # (6111,)
	s = np.sum(weights)
	print("s = ",s,np.random.rand(1)*s)  # s = 1.0
	sample = int(np.searchsorted(t, np.random.rand(1)*s))
	print("sample = ",sample)  # 随机数 从t中随机抽取
	# print("words = ",len(words))  # 6110
	return words[sample]   # 从诗歌汉子数据集中选取一个返回


w1 = to_word(probs,words)
w2 = get_word(probs,words)
print("w1 = ",w1,"w2 = ",w2)
index1 = words.index(w1)
index2 = words.index(w2)
print("index1 = ",index1,"index2 = ",index2)

