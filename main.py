import pandas as pd
import numpy as np
import Bug_data_processing as bdp
import Bug_word_embedding as bwe
import Extract_feature_bug as efb

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
# Convert txt to csv in order to select column data which we need to use
df = pd.read_csv('TestData\Eclipse_Platform_UI.txt', sep="\t")
df.to_csv('TestData\Eclipse_Platform_UI_csv.csv', index=False)

# Read file csv and Extract summary and description
inp = pd.read_csv('TestData\Eclipse_Platform_UI_csv.csv')
x= inp[['summary','description']]

# Data pre-processing
bdp.Start(x)

# Word_Embedding
matrixBug = bwe.MatrixOfBugReport()

# Extract feature vector
matrixs, maxSent = bwe.MatrixOfBugReport()
extractBug = efb.ExtractVector(matrixs,maxSent)

# # -------------------------------------------enhanceCNN---------------------------------------
# # input for enhance
# (ri_max1, fi_max1, ri_min1, fi_min1) = le1.labelEqual1(extractBug, extractSource)
# (ri_max2, fi_max2, ri_min2, fi_min2) = le0.computeCosine(extractBug, extractSource)
# r_max = max(ri_max1, ri_max2)
# f_max = max(fi_max1, fi_max2)
# r_min = min(ri_min1, ri_min2)
# f_min = min(fi_min1, fi_min2)
# gt.computeRiAndFi(r_max, f_max, r_min, f_min)
#
# # enhance CNN
# EnhanceCNN.enhance_CNN()