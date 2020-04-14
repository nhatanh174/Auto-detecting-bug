import os
import glob
import javalang
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
#xu ly word cu phap vd: anhNguyenNhat -> anh nguyen nhat
def process_word(word):

  char = []
  for i in range(0, len(word)):
    if word[i].islower() == True:
      char.append(word[i])
    else:
      char.append(' ')
      char.append(word[i].lower())
  return word_tokenize(''.join(char))

# function read folder
def openFolder(path, files, agr):
    files.extend(glob.glob(os.path.join(path, agr)))
    for file in os.listdir(path):
        fullpath = os.path.join(path, file)
        if os.path.isdir(fullpath) and not os.path.islink(fullpath):
            openFolder(fullpath,files,agr)
def preprocess_sourcefile():
    name_files=[]
    srs=[]
    openFolder(r"TestData/SourceFile/sourceFile_aspectj",name_files,"*.java")

    for name_file in name_files:
        source_code=[]
        with open(name_file,'r+') as f:
            sr=f.readlines()
        f.close()
        for line in sr:
            sentence=[]
            line = re.sub(r'\W', ' ', line)
            line = re.sub(r'\d', '', line)
            line = re.sub('_', ' ', line)
            if(line==''):
                continue
            tokens=list(javalang.tokenizer.tokenize(line))
            for i in range(len(tokens)):
                word=tokens[i].value
                if(isinstance(tokens[i],javalang.tokenizer.Identifier)):
                    if(len(word)==1):
                        continue
                    if(word.islower()==True):
                        sentence.append(word)
                        continue
                    if(word.isupper()==True):
                        sentence.append(word.lower())
                        continue
                    for ele in process_word(word):
                        if(len(ele)>1):
                            sentence.append(ele)
                else:
                    continue
                if (sentence==''):
                    continue
                else:
                    source_code.append(sentence)
        if(len(source_code)!=0):
            srs.append(source_code)
    return srs

preprocess_sourcefile()