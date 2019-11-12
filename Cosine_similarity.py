import  numpy as np
import math
import csv
import getDateAndFrequence as gt
import pandas as pd



def qs(cosine, l, r, tt):  # sort cosin to select 300 min cosine
    i = l
    j = r
    tg = cosine[int((l + r) / 2)]
    while (i <= j):
        while cosine[i] < tg:
            i += 1
        while cosine[j] > tg:
            j -= 1
        if i <= j:
            cosine[i], cosine[j] = cosine[j], cosine[i]
            tt[i], tt[j] = tt[j], tt[i]
            i += 1
            j -= 1
    if l < j:
        qs(cosine, l, j, tt)
    if i < r:
        qs(cosine, i, r, tt)


# -------------------------------------
def concatenate(a, b):
    x = np.zeros(shape=(2, 300))
    x[0] = a
    x[1] = b
    return x


# -------------------------------------

def computeCosine(bugVectors, sourceVectors):

    listName = gt.getListName()
    time_bug = gt.getListDateBug()
    time_source = gt.getListDateSource()
    dem = -1
    ri_max = 0
    fi_max = 0
    ri_min = 1000000
    fi_min = 1000000
    with open('Training_Test\InputForEnhanceCNN_label0.csv','w') as csv_file:
        fieldnames = ['input_vector','label','Ri','Fi']     #Định dạng cột
        writer = csv.DictWriter(csv_file,fieldnames=fieldnames)
        writer.writeheader()
        for bVector in bugVectors:
            list_cosine = []
            dem+=1
            stt = np.arange(len(sourceVectors))  # save stt of source file
            # compute cosine for each bug
            for sVector in sourceVectors:
                nume = np.sum(bVector * sVector)  # Numerator
                sqrt1 = math.sqrt(np.sum(bVector * bVector))
                sqrt2 = math.sqrt(np.sum(sVector * sVector))
                deno = sqrt1 * sqrt2
                list_cosine.append(nume / deno)
            # sort cosin
            qs(list_cosine, 0, len(list_cosine) - 1, stt)
            cout = 0
            for i in range(len(list_cosine)):
                cout+=1
                if cout==301:
                    break
                else:
                    concate_vector = concatenate(bVector,sourceVectors[stt[i]])

                    #compute Ri
                    ri, fi = gt.computeR_F(time_bug[dem],time_source,listName[stt[i]],listName)
                    ri_max = max(ri_max, ri)
                    fi_max = max(fi_max, fi)
                    ri_min = min(ri_min,ri)
                    fi_min = min(fi_min,fi)
                    writer.writerow({'input_vector':concate_vector, 'label':0, 'Ri':ri, 'Fi':fi}) #viết theo từng hàng
                # luu file csv voi cac cot la: vector ghep, id(not bug=0),Ri(date bug,date sorce), Frequence
                # nhung truoc do can luu date cua cac source file theo ten
                # cac source tuong ung voi bug trong bug report se co id(bug=1)
    return (ri_max, fi_max,ri_min, fi_min)

# tim cac cap vector, ghep lai, co label =1
