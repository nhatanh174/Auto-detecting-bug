
from keras.models import Model,Sequential
from keras.layers import Conv2D, MaxPooling2D,Input,Embedding, Dense, Flatten, BatchNormalization
import numpy as np
import pandas as pd
from keras import backend as K

def action():
    # enhance CNN with label(bug, notbug) and (r,f)
    def custom_loss(y_true, y_pred, r, f,w1,w2):
        return -K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred) - w1*r - w2*f)

    def enhance_loss(w1,w2,r, f):
        def loss(y_true, y_pred):
            return custom_loss(y_true, y_pred, r, f,w1,w2)

        return loss

    def returnRiFi():
        fi = pd.read_csv('Training_Test\InputForEnhanceCNN_label0.csv')
        r1 = fi['Ri'].values
        f1 = fi['Fi'].values
        r1 = r1.tolist()
        f1 = f1.tolist()
        fo = pd.read_csv('Training_Test\InputForEnhanceCNN_label1.csv')
        r2 = fo['Ri'].values
        f2 = fo['Fi'].values
        for i in r2:
            r1.append(i)
        x= np.asarray(r1)

        for i in f2:
            f1.append(i)
        y= np.asarray(f1)
        return x,y

    def returnLabel(count):
        fi = pd.read_csv('Training_Test\InputForEnhanceCNN_label1.csv')
        x = fi['label']
        x= len(x)
        arr = np.zeros(shape=[1,count])
        ind = count-x
        while (ind < count):
            arr[ind] =1
            ind+=1
        return arr

    # compute w1,w2
    def compute_w1_w2(r, f):
        w1 = []
        w2 = []
        w_init = np.random.normal(size=2)
        w1.append(w_init[0])
        w2.append(w_init[1])
        for i in range(1, len(r)):
            w1.append(w1[i - 1] - 0.001 * r[i - 1])
            w2.append(w2[i - 1] - 0.001 * f[i - 1])
        return w1, w2

    def enhance_CNN():

        r,f = returnRiFi()
        w1,w2= compute_w1_w2(r,f)
        count = len(r)
        label = returnLabel(count)
        model_CNN = Sequential()
        model_CNN.add(Conv2D(100, kernel_size=(2, 2), activation='relu', input_shape=input_vt.shape[1:]))
        model_CNN.add(BatchNormalization())
        model_CNN.add(MaxPooling2D(pool_size=(1, 2)))
        model_CNN.add(Flatten())
        model_CNN.add(Dense(1, activation='softmax'))
        model_CNN.compile(optimizer='adam', loss=enhance_loss(w1,w2,r, f), metrics=['accuracy'])
        model_CNN.fit(input_vt, label, epochs=100, batch_size=1, validation_data=(x_test, y_test))
