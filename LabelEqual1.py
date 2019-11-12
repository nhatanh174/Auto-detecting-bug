import pandas as pd
import getDateAndFrequence as gt
import numpy as np
import csv
import Cosine_similarity as cs

def concatenate(a, b):
    x = np.zeros(shape=(2, 300))
    x[0] = a
    x[1] = b
    return x


def labelEqual1(bugVectors, sourceVectors):
    listName = gt.getListName()
    time_bug = gt.getListDateBug()
    time_source = gt.getListDateSource()
    file = pd.read_csv('Training_Test\AspectJ_csv_trainingset.csv')
    data_des = file['description'].values
    id = file['commit'].values
    data_time = file['commit_timestamp'].values
    data_file1 = file['files'].values
    data_file2 = file['Unnamed:10'].values
    ri_max = 0
    fi_max = 0
    ri_min = 1000000
    fi_min = 1000000

    with open('Training_Test\InputForEnhanceCNN_label1.csv','w') as csv_file:
        fieldnames = ['input_vector','label','Ri','Fi']     #Định dạng cột
        writer = csv.DictWriter(csv_file,fieldnames=fieldnames)
        writer.writeheader()
        m= len(bugVectors)
        n = len(data_des)
        count_bug =0
        count = 0
        while (count_bug < m) and (count < n):
            if (type(data_des[count]) != str):
                count += 1
            else:
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



# sinh ngẫu nhiên vector, coi đó là vecto đặc trưng, để test code

bugvt = []
sourvt =[]
for i in range(382):
    x= np.random.rand(1,300)
    bugvt.append(x)
for i in range(2394):
    x = np.random.rand(1, 300)
    sourvt.append(x)
(ri_max1,fi_max1,ri_min1,fi_min1) = labelEqual1(bugvt,sourvt)
(ri_max2, fi_max2,ri_min2, fi_min2) = cs.computeCosine(bugvt,sourvt)
r_max = max(ri_max1,ri_max2)
f_max = max(fi_max1, fi_max2)
r_min = min(ri_min1,ri_min2)
f_min = min (fi_min1,fi_min2)
gt.computeRiAndFi(r_max,f_max,r_min,f_min)



