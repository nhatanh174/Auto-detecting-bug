import pandas as pd
import numpy as np
import  pickle
import re
import Bug_data_processing as bdp
import Bug_word_embedding as bwe
import Extract_feature_bug as efb

# for source file
from sklearn.model_selection import train_test_split
import Source_data_processing as sdp
import Source_word2vec
import Source_tfidf
import Extract_feature_source

# setup input for enhance
import Label_equal0 as le0
import Label_equal1 as le1
import Get_date_frequence as gt
import EnhanceCNN

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

pickle_input = open("TestData/Bug_extract_feature.pickle", "rb")
listt = pickle.load(pickle_input)
pickle_input.close()

extractBug = np.asarray(listt)
#----------------------------
file = pd.read_csv('TestData/Sourcefiles.csv')
x = file['source_file'].values
vector =[]
for i in x:
    mn = to_matrix(i, 1, 300)
    vector.append(mn)
extractSource = np.asarray(vector)

(ri_max1, fi_max1, ri_min1, fi_min1) = le1.labelEqual1(extractBug, extractSource)
(ri_max2, fi_max2, ri_min2, fi_min2) = le0.computeCosine(extractBug, extractSource)
r_max = max(ri_max1, ri_max2)
f_max = max(fi_max1, fi_max2)
r_min = min(ri_min1, ri_min2)
f_min = min(fi_min1, fi_min2)
gt.computeRiAndFi(r_max, f_max, r_min, f_min)

