import pandas as pd
import Data_preprocessing as pre_bug
import Bug_report_processed as brp
import Extract_feature_vector as efv
import numpy as np

#for source file
import File as file
import Data_processing as pre_sour   #process source
import Extract_vecto as ex_vec
import model_CNN
import Word2vec_source as w2vs
import TF_IDF as ti

import Cosine_similarity as cs
from sklearn.model_selection import train_test_split

#------------------------------bug report-----------------------------------
#
# df = pd.read_csv('Training_Test\AspectJ.txt', sep="\t")
# df.to_csv('Training_Test\AspectJ_csv.csv', index=False)
#
#
#
# # split into training set, validation set, test set
# inp = pd.read_csv("Training_Test\AspectJ_csv.csv")
# x= inp[['id', 'bug_id', 'summary', 'description', 'report_time', 'report_timestamp', 'status', 'commit', 'commit_timestamp', 'files', 'Unnamed: 10']].values
#
# y=np.arange(len(x))
# columns = "id bug_id summary description report_time report_timestamp status commit commit_timestamp files Unnamed:10".split()
# X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
# df =pd.DataFrame(X_train,columns=columns)
# df.to_csv("Training_Test\AspectJ_csv_trainingset.csv")
# #-----------------------------------------------------------------------------------------------
#
# # Read file csv and Extract summary and description
# inp = pd.read_csv('Training_Test\AspectJ_csv_trainingset.csv')
# x= inp[['summary','description']]
#
# df =pd.DataFrame(x)
#
# df.to_csv('Training_Test\AspectJ_process.csv')
#
# #Data pre-processing
# pre_bug.Start()
#
# #Word_Embedding
# brp.word_embedding()
#
# #Extract feature vector
# vecto_detect_bug = efv.Start()
# print(vecto_detect_bug)

#---------------for source file--------------


path= r"E:\Pycharm projects\Auto_detect_bug\Training_Test\New folder"

files=[]        # list name file.java
file.openFolder(path,files,'*.java')
# list sourfile
corpus=[]
for file in files:
    corpus.append(pre_sour.pre_processing(file))
corpus_2d=[]
for file in corpus:
    for sent in file:
        corpus_2d.append(sent)

#word2vec
w2vs.start(corpus_2d)

# list matrix of sourfiles
number_line=[]
for sourcefile in corpus:
    number_line.append(len(sourcefile))
line_max= max(number_line)
matrix=[]
tf_idf = ti.computeTf_Idf(corpus)

print(corpus)
x= np.asarray(corpus)
print(x.shape)

for sourcefile in corpus:
    vecto=np.zeros(shape=(line_max,300))
    for i in range(len(sourcefile)):
        vecto[i]= ex_vec.CombineVector(sourcefile[i],tf_idf)
        print(i)
    matrix.append(vecto)
x= np.asarray(matrix)
print(x.shape)
# # #matrix là list các ma trận của các source file
# # #list vecto feature
# # vecto_detect_source=model_CNN.feature_detect(matrix)
# # #print(vecto_detect_bug)
# # print(len(vecto_detect_source))
# # # cs.computeCosine(vecto_detect_bug,vecto_detect_source)

#-------------------------------------------enhanceCNN---------------------------------------