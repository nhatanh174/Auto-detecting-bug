import File
import pandas as pd
import os
import numpy as np
from datetime import datetime

# Define constance
BUG="TestData\Eclipse_Platform_UI.txt"
BUG_CSV ="TestData\Eclipse_Platform_UI_csv.csv"
BUG_PROCESS = "TestData\Eclipse_Platform_UI_process.csv"
SOURCE = "TestData/SourceFile/sourceFile_eclipseUI"

# get the list of file source name
def getListName():
    path_source = SOURCE
    files = []  # list name file.java
    File.openFolder(path_source, files, '*.java')
    ds = File.getName(files)
    return ds


# -------------------------------------------------
# xử lý link trong cột file, lấy ra tên file
def divide_link(input):
    output = []
    link = ''
    for i in range(len(input)):
        if input[i] == ' ' and input[i - 5:i] == ".java":
            output.append(link)
            link = ''
        else:
            link += input[i]
    output.append(link)
    return output

def clean(link, id):  # add id , return then id+name file java
    path = os.path.normpath(link)
    token = path.split(os.sep)
    return (id + ' ' + token[len(token) - 1])

def listFileInBug(link,id):
    s=[]
    list_link = divide_link(link)
    for i in list_link:
        s.append(clean(i,id))
    return s

# get list of the date of bug
def getListDateBug():
    path_bug = BUG_CSV
    file = pd.read_csv(path_bug)
    des = file['description']
    date = file['report_timestamp']
    n = len(des)
    count= 0
    listDate=[]
    while count <n:
        listDate.append(date[count])
        count+=1
    return listDate

#get list of the date of file
def getListDateSource():
    file = pd.read_csv(BUG_CSV)
    data_des=file['description'].values
    id = file['commit'].values
    data_time = file['commit_timestamp'].values
    data_file1 = file['files'].values
    data_file2 = file['Unnamed:10'].values
    n= len(data_des)
    listName=getListName()
    m=len(listName)
    count = 0
    listDate = np.zeros(m)
    while count<n:
        s = listFileInBug(data_file1[count],id[count])
        for i in s:
            ind = listName.index(i)
            listDate[ind]=data_time[count]
        if (type(data_file2[count])==str):
            r = listFileInBug(data_file2[count], id[count])
            for i in r:
                ind = listName.index(i)
                listDate[ind] = data_time[count]
        count+=1
    return listDate.tolist()

#Compute Ri, Fi
def tokenize_name(data):
    data=data.split()
    return data[1]

#convert timestamp to date
def fromtimestamp(x):
    readable = datetime.fromtimestamp(x).isoformat()
    datetimeObj = datetime.strptime(readable, '%Y-%m-%dT%H:%M:%S')  # '2018-09-11T15::11::45.456777'
    return (datetimeObj.month, datetimeObj.year)

def sub(a1,b1, a2,b2):
    if (b1==b2):
        return a1-a2
    else:
        return 12*(b1-b2)+a1-a2
def computeR_F(dateBug, listDateSource,nameSource,listNameSource):
    name = tokenize_name(nameSource)
    count =0
    ri=0
    f=0
    lim = len(listNameSource)
    while count<lim:
        if (name in listNameSource[count]) and (listDateSource[count]<= dateBug):
            f+=1
            if listDateSource[count]>ri:
                ri = listDateSource[count]
        count+=1
    #print(fromtimestamp(dateBug), fromtimestamp(ri))
    n1,m1 = fromtimestamp(dateBug)
    n2,m2 = fromtimestamp(ri)
    r = 1/(sub(n1,m1,n2,m2) +1)
    return (r,f)

#-------------------------------------
##compute Ri,Fi
def computeRiAndFi(rMax, fMax,rMin, fMin):
    file = pd.read_csv('TestData/InputForEnhanceCNN_label0.csv')
    data_ri = file['Ri']
    data_fi = file['Fi']
    ri =[]
    fi =[]
    for i in data_ri:
        if i< rMin:
            ri.append(0)
        else:
            if rMin <= i and i<=rMax:
                x = (i-rMin)/(rMax-rMin)
                ri.append(x)
            else:
                ri.append(1)
    for i in data_fi:
        if i<fMin:
            fi.append(0)
        else:
            if fMin <= i and i <= fMax:
                x= (i-fMin)/(fMax-fMin)
                fi.append(x)
            else:
                fi.append(1)
    file['Ri'] = ri
    file['Fi'] = fi
    file.to_csv('TestData/InputForEnhanceCNN_label0.csv')

    file = pd.read_csv('TestData/InputForEnhanceCNN_label1.csv')
    data_ri = file['Ri']
    data_fi = file['Fi']
    ri = []
    fi = []
    for i in data_ri:
        if i < rMin:
            ri.append(0)
        else:
            if rMin <= i and i <= rMax:
                x = (i - rMin) / (rMax - rMin)
                ri.append(x)
            else:
                ri.append(1)
    for i in data_fi:
        if i < fMin:
            fi.append(0)
        else:
            if fMin <= i and i <= fMax:
                x = (i - fMin) / (fMax - fMin)
                fi.append(x)
            else:
                fi.append(1)
    file['Ri'] = ri
    file['Fi'] = fi
    file.to_csv('TestData/InputForEnhanceCNN_label1.csv')


