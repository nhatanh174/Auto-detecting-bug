import pandas as pd
import Get_date_frequence as gt
import numpy as np
import csv

import re


def concatenate(a, b):
    x = np.zeros(shape=(2, 300))
    x[0] = a
    x[1] = b
    return x


def labelEqual1(bugVectors, sourceVectors):
    listName = gt.getListName()
    time_bug = gt.getListDateBug()
    time_source = gt.getListDateSource()
    file = pd.read_csv('TestData/AspectJ_csv.csv')
    data_des = file['description'].values
    id = file['commit'].values
    data_time = file['commit_timestamp'].values
    data_file1 = file['files'].values
    data_file2 = file['Unnamed:10'].values
    ri_max = 0
    fi_max = 0
    ri_min = 1000000
    fi_min = 1000000

    with open('TestData/InputForEnhanceCNN_label1.csv','w') as csv_file:
        fieldnames = ['input_vector','label','Ri','Fi']     #Định dạng cột
        writer = csv.DictWriter(csv_file,fieldnames=fieldnames)
        writer.writeheader()
        m= len(bugVectors)
        n = len(data_des)
        count_bug =0
        count = 0
        while (count_bug < m) and (count < n):
            s = gt.listFileInBug(data_file1[count], id[count])
            for i in s:
                ind = listName.index(i)
                concate_vector = concatenate(bugVectors[count_bug],sourceVectors[ind])
                ri, fi = gt.computeR_F(time_bug[count_bug], time_source, listName[ind], listName)
                writer.writerow({'input_vector':concate_vector, 'label':1, 'Ri':ri, 'Fi':fi}) #viết theo từng hàng
            if (type(data_file2[count]) == str):
                r = gt.listFileInBug(data_file2[count], id[count])
                for i in r:
                    ind = listName.index(i)
                    ri, fi = gt.computeR_F(time_bug[count_bug], time_source, listName[ind], listName)
                    ri_max = max(ri_max, ri)
                    fi_max = max(fi_max, fi)
                    ri_min = min(ri_min, ri)
                    fi_min = min(fi_min, fi)
                    concate_vector = concatenate(bugVectors[count_bug], sourceVectors[ind])
                    writer.writerow({'input_vector': concate_vector, 'label': 1, 'Ri': ri, 'Fi': fi})  # viết theo từng hàng
            count_bug =count_bug+1
            count = count+1
    return (ri_max, fi_max,ri_min, fi_min)

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




