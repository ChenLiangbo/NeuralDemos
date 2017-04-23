#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
 
image_dir = 'girls'
new_girl_dir = 'little_girls'
if not os.path.exists(new_girl_dir):
    os.makedirs(new_girl_dir)
 
for img_file in os.listdir(image_dir):
    img_file_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_file_path)
    if img is None:
        print("image read fail")
        continue
    height, weight, channel = img.shape
    if height < 200 or weight < 200 or channel != 3: 
        continue
    # 你也可以转为灰度图片(channel=1)，加快训练速度
    # 把图片缩放为64x64
    img = cv2.resize(img, (64, 64))
    new_file = os.path.join(new_girl_dir, img_file)
    cv2.imwrite(new_file, img)
    print(new_file)