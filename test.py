
# coding: utf-8

# In[1]:

import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
#import matplotlib.pyplot as plt
import csv


# In[2]:

lines = tf.gfile.GFile('F:/TianChiXueLang/cloth_detection0720/tmp/output_labels3000.txt').readlines()
uid_to_human = {}
#一行一行读取数据
for uid,line in enumerate(lines) :
    #去掉换行符
    line=line.strip('\n')
    uid_to_human[uid] = line

def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]


#创建一个图来存放google训练好的模型
with tf.gfile.FastGFile('F:/TianChiXueLang/cloth_detection0720/tmp/output_graph3000.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
headers = ['filename', 'probability']

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    #遍历目录
    csv_file=open("F:/TianChiXueLang/cloth_detection0720/step3000rate0.01.csv",'w',newline='')
    write_csv=csv.writer(csv_file)
    write_csv.writerow(('filename','probability'))              #将测试结果打印至.CSV文件中，然后提交
    # for i in range(10):
    #     write_csv.writerow((i,i * 2))


    for root,dirs,files in os.walk('F:/TianChiXueLang/test_data/'):
        for file in files:
            #载入图片
            print('>>>>>>>>>>>>>>>>>>>>>>')
            print('file=',file)
            image_data = tf.gfile.FastGFile(os.path.join(root,file), 'rb').read()
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})#图片格式是jpg格式
            predictions = np.squeeze(predictions)#把结果转为1维数据
            #write_csv.writerow(('filename', 'probability'))
            #print('prediction=',predictions)

            print('faultscore=',predictions[1])
            probability = predictions[1]
            probability="%.6f"%probability          #控制精度
            write_csv.writerow((file,probability))
            #打印图片路径及名称
            # image_path = os.path.join(root,file)
            # print(image_path)
            # #显示图片
            # img=Image.open(image_path)
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()

            #排序
            top_k = predictions.argsort()[::-1]
            #top_k=predictions
            print(top_k)
            for node_id in top_k:     
                #获取分类名称
                print('top_k=',top_k)
                human_string = id_to_string(node_id)
                #获取该分类的置信度
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print()






