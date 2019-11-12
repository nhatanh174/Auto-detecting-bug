# from threading import Thread
# import threading
# import time
#
#
# def cal_square(numbers):
# 	print("calculate square number")
# 	for n in numbers:
# 		time.sleep(0.2)
# 		print ('square:', n*n)
#
#
#
# def cal_cube(numbers):
# 	print("calculate cube number \n")
# 	for n in numbers:
# 		time.sleep(0.2)
# 		print ('cube:', n*n*n)
#
#
# arr = [2,3,7,9]
#
# try:
# 	t = time.time()
# 	t1 = threading.Thread(target=cal_square, args=(arr,))
# 	t2 = threading.Thread(target=cal_cube, args=(arr,))
#
# 	t1.start()
# 	t2.start()
# 	t1.join()
# 	t2.join()
#
# 	print ("done in ", time.time()- t)
# except:
# 	print ("error")

import numpy as np
import pandas as pd
import csv
import re
import File as file
from datetime import datetime
#
# with open('Training_Test\\test.csv', 'w') as csv_file:
#     fieldnames = ['input_vector', 'label']  # Định dạng cột
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     writer.writeheader()
#     for i in range(10):
#         concate_vector = np.random.rand(10)
#         writer.writerow({'input_vector': concate_vector, 'label': 0})
# fi = pd.read_csv('Training_Test\\test.csv')
# m=[]
# for i in range(10):
#     m.append(i)
# fi['label'] =m
# fi.to_csv('Training_Test\\test.csv')
#
# def returnRi():
#     fi = pd.read_csv('Training_Test\InputForEnhanceCNN_label0.csv')
#     r1 = fi['Ri'].values
#     f1 = fi['Fi'].values
#     r1 = r1.tolist()
#     f1 = f1.tolist()
#     fo = pd.read_csv('Training_Test\InputForEnhanceCNN_label1.csv')
#     r2 = fo['Ri'].values
#     f2 = fo['Fi'].values
#     for i in r2:
#         r1.append(i)
#     x = np.asarray(r1)
#
#     for i in f2:
#         f1.append(i)
#     y = np.asarray(f1)
#     return x, y
# m,n = returnRi()
# print(type(m))

def to_matrix(input,shape1,shape2):
    matrix = np.zeros(shape=(shape1,shape2))
    mtrx = re.sub("\[", ' ', input)
    mtrx = re.sub("]", ' ', mtrx)
    mtrx = mtrx.split()
    count_col = 0
    count_row = 0
    for i in mtrx:
        j = float(i)
        if (count_col<shape2):
            matrix[count_row][count_col]=j
        if (count_col==shape2) and (count_row<shape1):
            count_col=0
            count_row+=1
            matrix[count_row][count_col] = j
        count_col += 1
    return matrix.tolist()

file = pd.read_csv('Training_Test\InputForEnhanceCNN_label0.csv')
x = file['input_vector'].values
for i in x[:5]:
    mn = to_matrix(i, 2,300)
    print(type(mn))

    for j in mn:
        print(j)
    print(mn)

