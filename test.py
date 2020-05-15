import pandas as pd
import Get_date_frequence as gt
import File
import os
import glob
import ntpath
import numpy as np
import pickle

# # Define constance
# BUG="TestData\Eclipse_Platform_UI.txt"
# BUG_CSV ="TestData\Eclipse_Platform_UI_csv.csv"
# BUG_PROCESS = "TestData\Eclipse_Platform_UI_process.csv"
# SOURCE = "TestData/SourceFile/sourceFile_eclipseUI"
#
# # xử lý link trong cột file, lấy ra tên file
# def divide_link(input):
#     output = []
#     link = ''
#     # for i in input:
#     #     if (i != ' '):
#     #         link += i
#     #     else:
#     #         output.append(link)
#     #         link = ''
#     # output.append(link)
#     for i in range(len(input)):
#         if input[i]==' ' and input[i-5:i]==".java":
#             output.append(link)
#             link=''
#         else:
#             link+=input[i]
#     output.append(link)
#     return output
#
# def clean(link, id):  # add id , return then id+name file java
#     path = os.path.normpath(link)
#     token = path.split(os.sep)
#     return (id + ' ' + token[len(token) - 1])
#
# def listFileInBug(link,id):
#     s=[]
#     list_link = divide_link(link)
#     for i in list_link:
#         s.append(clean(i,id))
#     return s
#
# def openFolder(path, files, agr):
#     files.extend(glob.glob(os.path.join(path, agr)))
#     for file in os.listdir(path):
#         fullpath = os.path.join(path, file)
#         if os.path.isdir(fullpath) and not os.path.islink(fullpath):
#             openFolder(fullpath,files,agr)
#
# def getName(files):
#     for i in range(0,len(files)):
#         files[i]=ntpath.basename(files[i])
#     return files
#
# file = pd.read_csv(BUG_CSV)
# data_des = file['description'].values
# id = file['commit'].values
# data_time = file['commit_timestamp'].values
# data_file1 = file['files'].values
# data_file2 = file['Unnamed: 10'].values
# n = len(data_des)
# count=0
# re=[]
# while (count < n):
#     s = listFileInBug(data_file1[count], id[count])
#     for i in s:
#         re.append(i)
#     if (type(data_file2[count]) == str):
#         r = listFileInBug(data_file2[count], id[count])
#         for i in r:
#             re.append(i)
#     count+=1
#
# # for i in re:
# #     print(i)
# print(len(re))
#
# name_files=[]
# openFolder(SOURCE,name_files,"*.java")
# file = getName(name_files)
#
# print(len(name_files))
# sourceFileBug =[]
# for i in re:
#     kt=0
#     for j in file:
#         if (i==j):
#             kt=1
#             break
#     if kt==1:
#         sourceFileBug.append(i)
# print(len(sourceFileBug))
# dem=0
# for i in file:
#     kt=0
#     for j in re:
#         if (i==j):
#             kt=1
#             break
#     if kt==0:
#         print(i)
#         dem+=1
# print(dem)
#
# pickle_input = open("TestData/new_vectorsource.pickle", "rb")
# listt = pickle.load(pickle_input)
# pickle_input.close()
# extractSource = np.asarray(listt)
# print(extractSource.shape)

from gensim.models import Word2Vec
model = Word2Vec.load('w2v_summary.model')
word=""
print(model.wv[word])