# import numpy as np
# import os
# import sys
# import glob
# import argparse
#
# src_path='F:/TianChiXueLang/xuelang_round1_train_part3_20180709/fault'
# newpath='F:/TianChiXueLang/datatest'
# jpg_list = glob.glob(src_path + '/*/*.jpg' )
# for i in jpg_list:
#     full_path = jpg_list[i]  # 将文件名与文件目录连接起来，形成完整路径
#     des_path = newpath + '/' + name  # 目标路径，将该文件夹信息添加进最后的文件名中
#     if filename in name:  # 匹配符合条件的文件，也可用if(name.find(filename)!=-1):
#         shutil.move(full_path, des_path)  # 移动文件到目标路径（移动+重命名）
# print('jpg_list=',jpg_list)

import os
import shutil
import glob

#floderaddress = 'F:/TianChiXueLang/xuelang_round1_train_part3_20180709/fault'
#floderaddress='F:/TianChiXueLang/xuelang_round1_train_part2_20180705/fault1'
floderaddress='F:/TianChiXueLang/xuelang_round1_train_part1_20180628/fault'
dest_dir='F:/TianChiXueLang/cloth_traindata/fault'
file_all =  glob.glob(floderaddress + '/*/*.jpg' )
for name in file_all:
    shutil.copy(name, dest_dir)



