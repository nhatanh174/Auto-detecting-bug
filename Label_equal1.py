import pandas as pd
import Get_date_frequence as gt
import numpy as np
import csv
import pickle
import re

# Define constance
BUG="TestData\Eclipse_Platform_UI.txt"
BUG_CSV ="TestData\Eclipse_Platform_UI_csv.csv"
BUG_PROCESS = "TestData\Eclipse_Platform_UI_process.csv"
SOURCE = "TestData/SourceFile/sourceFile_eclipseUI"

def concatenate(a, b):
    x = np.zeros(shape=(2, 300))
    x[0] = a
    x[1] = b
    return x



def labelEqual1(bugVectors, sourceVectors,matrixBug,matrixSource):
    listName = gt.getListName()
    time_bug = gt.getListDateBug()
    time_source = gt.getListDateSource()
    file = pd.read_csv(BUG_CSV)
    data_des = file['description'].values
    id = file['commit'].values
    data_time = file['commit_timestamp'].values
    data_file1 = file['files'].values
    data_file2 = file['Unnamed: 10'].values
    ri_max = 0
    fi_max = 0
    ri_min = 1000000
    fi_min = 1000000

    with open('TestData/InputForEnhanceCNN_label1.csv','w') as csv_file:
        fieldnames = ['matrixBug','matrixSource','label','Ri','Fi']     #Định dạng cột
        writer = csv.DictWriter(csv_file,fieldnames=fieldnames)
        writer.writeheader()
        m= len(bugVectors)
        n = len(data_des)
        count_bug =0
        count = 0
        while (count_bug < m) and (count < n):
            s = gt.listFileInBug(data_file1[count], id[count])
            s1=[]
            for i in s:
                kt=0
                for j in listName:
                    if (i==j):
                        kt=1
                        break
                if kt==1:
                    s1.append(i)        # loai tru cac file k can thiet ma tac gia da xoa di
            s=s1
            for i in s:
                ind = listName.index(i)
                # concate_vector = concatenate(bugVectors[count_bug],sourceVectors[ind])
                ri, fi = gt.computeR_F(time_bug[count_bug], time_source, listName[ind], listName)
                # writer.writerow({'input_vector':concate_vector, 'label':1, 'Ri':ri, 'Fi':fi}) #viết theo từng hàng
                writer.writerow({'matrixBug': matrixBug[count_bug], 'matrixSource': matrixSource[ind], 'label': 1, 'Ri': ri,'Fi': fi})  # viết theo từng hàng
            if (type(data_file2[count]) == str):
                r = gt.listFileInBug(data_file2[count], id[count])
                r1 = []
                for i in r:
                    kt = 0
                    for j in listName:
                        if (i == j):
                            kt = 1
                            break
                    if kt == 1:
                        r1.append(i)  # loai tru cac file k can thiet ma tac gia da xoa di
                r=r1
                for i in r:
                    ind = listName.index(i)
                    ri, fi = gt.computeR_F(time_bug[count_bug], time_source, listName[ind], listName)
                    ri_max = max(ri_max, ri)
                    fi_max = max(fi_max, fi)
                    ri_min = min(ri_min, ri)
                    fi_min = min(fi_min, fi)
                    # concate_vector = concatenate(bugVectors[count_bug], sourceVectors[ind])
                    # writer.writerow({'input_vector': concate_vector, 'label': 1, 'Ri': ri, 'Fi': fi})  # viết theo từng hàng
                    writer.writerow({'matrixBug':matrixBug[count_bug],'matrixSource':matrixSource[ind], 'label': 1, 'Ri': ri, 'Fi': fi})  # viết theo từng hàng
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




