import pandas as pd
import numpy as np
import pickle
import re
import Bug_data_processing as bdp
import Bug_word_embedding as bwe
import Extract_feature_bug as efb
import Label_equal1 as le1
import Label_equal0 as le0
import Get_date_frequence as gt

# Define constance
BUG="TestData\Eclipse_Platform_UI.txt"
BUG_CSV ="TestData\Eclipse_Platform_UI_csv.csv"
BUG_PROCESS = "TestData\Eclipse_Platform_UI_process.csv"

# # for source file
# from sklearn.model_selection import train_test_split
# import Source_data_processing as sdp
# import Source_word2vec
# import Source_tfidf
# import Extract_feature_source
#
# # setup input for enhance
# import Label_equal0 as le0
# import Label_equal1 as le1
# import Get_date_frequence as gt
# import EnhanceCNN



# ---------------for source file--------------
# srs = sdp.preprocess_sourcefile()
#
# Source_word2vec.start(srs)
# # compute tf_idf
# tf_idf=Source_tfidf.computeTf_Idf(srs)
# # extract vector with CNN
# extractSource = Extract_feature_source.sourcefile_extractvector(srs,tf_idf)

# ------------------------------bug report-----------------------------------
# # Convert txt to csv in order to select column data which we need to use
# df = pd.read_csv(BUG, sep="\t")
# df.to_csv(BUG_CSV, index=False)
#
# # Read file csv and Extract summary and description
# inp = pd.read_csv(BUG_CSV)
# x= inp[['summary','description']]
#
# # Data pre-processing
# bdp.Start(x)
# print("1")
# # Extract feature vector
# matrixs, maxSent = bwe.MatrixOfBugReport()
# print("2")
# extractBug = efb.ExtractVector(matrixs,maxSent)
# print("3")

# -------------------------------------------enhanceCNN---------------------------------------
def to_matrix(input, shape1, shape2):
    matrix = np.zeros(shape=(shape1, shape2))
    mtrx = re.sub("\[", ' ', input)
    mtrx = re.sub("]", ' ', mtrx)
    mtrx = mtrx.split()
    count_col = 0
    count_row = 0
    for i in mtrx:
        j = float(i)
        if (count_col < shape2):
            matrix[count_row][count_col] = j
        if (count_col == shape2) and (count_row < shape1):
            count_col = 0
            count_row += 1
            matrix[count_row][count_col] = j
        count_col += 1
    return matrix.tolist()
pickle_input = open("TestData/Bug_extract_feature.txt", "rb")
listt = pickle.load(pickle_input)
pickle_input.close()

matrixBug = np.asarray(listt)
vectorBug=[]
for matrix in matrixBug:
    vectorBug.append(np.mean(matrix))
#----------------------------
pickle_input = open("TestData/new_vectorsource.pickle", "rb")
listt = pickle.load(pickle_input)
pickle_input.close()
matrixSource = np.asarray(listt)
vectorSource=[]
for vector in vectorSource:
    vectorSource.append(np.mean(vector))

# input for enhance
(ri_max1, fi_max1, ri_min1, fi_min1) = le1.labelEqual1(matrixBug, matrixSource,vectorBug,vectorSource)
(ri_max2, fi_max2, ri_min2, fi_min2) = le0.computeCosine(matrixBug, matrixSource,vectorBug,vectorSource)
r_max = max(ri_max1, ri_max2)
f_max = max(fi_max1, fi_max2)                                      
r_min = min(ri_min1, ri_min2)
f_min = min(fi_min1, fi_min2)
gt.computeRiAndFi(r_max, f_max, r_min, f_min)

# enhance CNN
# EnhanceCNN.enhance_CNN()
