import gensim
from gensim.models import Word2Vec,Doc2Vec
import pandas as pd
import numpy as np
from nltk import sent_tokenize,word_tokenize

# Define constance
BUG="TestData\Eclipse_Platform_UI.txt"
BUG_CSV ="TestData\Eclipse_Platform_UI_csv.csv"
BUG_PROCESS = "TestData\Eclipse_Platform_UI_process.csv"

def MatrixOfBugReport():
    # compute the max sentences in description
    def max_sentence(data):
        max = 0
        for row in data:
            if (type(row) == str):
                sen = sent_tokenize(row)
                if (len(sen) > max):
                    max = len(sen)
        return max

    # split word in each sentence in summary
    def inputW2v(data):
        result = []
        for i in data:
            bugReport = i.split()
            result.append(bugReport)
        return result

    # word embedding summary, use word2vec, summary always contain one sentence
    def forSummary(text):
        model = Word2Vec.load('w2v_summary.model')
        summaryMatrix = []
        for line in text:
            sum = 0
            vector = np.zeros(shape=(1,300))
            for word in line:
                sum+=1
                vector+=model.wv[word]
            summaryMatrix.append(vector/sum)
        del model
        return np.asanyarray(summaryMatrix)

    # word embedding description, use sent2vec, description contain many sentences
    def forDescription(text, maxSent):
        model= Doc2Vec.load('Doc2vec.model')
        model.init_sims(replace=True)
        result =[]
        k=300
        for row in text:
            if type(row) ==str:
                sentences = sent_tokenize(row)
                vector = np.zeros(shape=(maxSent,k))
                cout =-1
                for sentence in sentences:
                    cout+=1
                    word = word_tokenize(sentence.lower())
                    vector[cout]=model.infer_vector(word)
                result.append(vector)
            else:
                vector = np.zeros(shape=(maxSent,k))
                result.append(vector)
        return np.asarray(result)
    inp = pd.read_csv(BUG_PROCESS)
    summary = inp['summary'].values
    text = inputW2v(summary)
    summaryMatrix = forSummary(text)
    description = inp['description'].values
    maxSent = max_sentence(description)
    descriptionMatrix = forDescription(description,maxSent)
    # print(summaryMatrix.shape, descriptionMatrix.shape)
    # Concatenate vector summary and matrix bug
    bugMatrix =[]
    for i in range(len(summaryMatrix)):
        bugMatrix.append(np.concatenate((summaryMatrix[i], descriptionMatrix[i])))

    print(len(bugMatrix),len(bugMatrix[1]),"ok")
    bugMatrix=np.asarray(bugMatrix)
    return  bugMatrix, maxSent+1


