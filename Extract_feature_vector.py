import csv
from gensim.models import Doc2Vec
from nltk import sent_tokenize,word_tokenize
import numpy as np
import pandas as pd

from keras.models import Model,Sequential
from keras.layers import Conv1D, MaxPooling1D,Input,Embedding

def Start():
    file = pd.read_csv('Training_Test\AspectJ_process.csv')
    data = file['description'].values

    # Find the maximum number (nS) of sentences from each bug report
    def max_sentence(data):
        max = 0
        for row in data:
            if (type(row) == str):
                sen = sent_tokenize(row)
                if (len(sen) > max):
                    max = len(sen)
        return max

    def equal(array, x, idx):
        array[idx] = x
        # return array

    k = 300
    max_sent = max_sentence(data)
    model = Doc2Vec.load('Doc2vec.model')
    feature_vector = []  # save feature vectors in bug reports
    #---------------------------------------------------------------

    for row in data:
        if (type(row) == str):
            sent = sent_tokenize(row)
            vector = np.zeros(shape=(max_sent, k))
            cout = 0
            for i in sent:
                word_data = word_tokenize(i.lower())
                vector[cout] = model.infer_vector(word_data)
                cout += 1

            # Build CNN model---------------------------------------------------------

            # Add model layers

            fil_sizes = [2, 3, 4]
            batch = np.expand_dims(vector, axis=0)
            # f_vector = np.zeros(shape=(1,3,100))

            i = 0
            arr = np.zeros(shape=(3, 100))
            dem = 0
            for filter_sizes in fil_sizes:
                model_CNN = Sequential()
                model_CNN.add(Conv1D(100, filter_sizes, activation='relu', name='conv1d', input_shape=vector.shape))
                model_CNN.add(MaxPooling1D(max_sent - filter_sizes + 1))
                model_CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                conv_vector = model_CNN.predict(batch)
                for i in conv_vector:
                    for j in i:
                        equal(arr, j, dem)  # giam so chieu vector
                dem += 1
            arr = arr.reshape(1, 300)
            feature_vector.append(arr)
    return feature_vector   # trả về các vector đặc trưng đối vời từng bug report
    # for i in feature_vector:
    #     print(i)


