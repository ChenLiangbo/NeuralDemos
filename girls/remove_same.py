#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
 
# 判断两张图片是否完全一样（使用哈希应该要快很多）
def is_same_image(img_file1, img_file2):
    img1 = cv2.imread(img_file1)
    img2 = cv2.imread(img_file2)
    if img1 is None or img2 is None:
        return False
    if img1.shape == img2.shape and not (np.bitwise_xor(img1, img2).any()):
        return True
    else:
        return False
 
# 去除重复图片
file_list = os.listdir('little_girls')
try:
	for img1 in file_list:
		print(len(file_list))
		for img2 in file_list:
			if img1 != img2:
				if is_same_image('little_girls/'+img1, 'little_girls/'+img2) is True:
					print(img1, img2)
					os.remove('little_girls/'+img1)
		file_list.remove(img1)
except Exception as e:
	print(e)