#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import os

# http://blog.csdn.net/u014365862/article/details/53868557
# 数据集
# https://pan.baidu.com/s/1kVSA8z9 (密码: atqm)
# https://pan.baidu.com/s/1ctbd9O (密码: kubu)
# 下载的图片分布在多个目录，把图片汇总到一个新目录
old_dir = 'images'
new_dir = 'girls'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
 
count = 0
for (dirpath, dirnames, filenames) in os.walk(old_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            new_filename = str(count) + '.jpg'
            os.rename(os.sep.join([dirpath, filename]), os.sep.join([new_dir, new_filename]))
            print(os.sep.join([dirpath, filename]))
            count += 1
print("Total Picture: ", count)

