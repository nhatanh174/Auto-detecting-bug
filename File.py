import os
import glob
import datetime
import ntpath

# function read folder
def openFolder(path, files, agr):
    files.extend(glob.glob(os.path.join(path, agr)))
    for file in os.listdir(path):
        fullpath = os.path.join(path, file)
        if os.path.isdir(fullpath) and not os.path.islink(fullpath):
            openFolder(fullpath,files,agr)
# get month and year of
def getMonth(file):
    t = os.path.getmtime(file)
    a = datetime.datetime.fromtimestamp(t)
    return (a.date().month,a.date().year)
# get name of file
def getName(files):
    for i in range(0,len(files)):
        files[i]=ntpath.basename(files[i])
    return files