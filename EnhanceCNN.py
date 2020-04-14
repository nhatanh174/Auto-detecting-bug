import re
from keras.models import Model,Sequential
from keras.layers import Conv2D, MaxPooling2D,Input,Embedding, Dense, Flatten, BatchNormalization, Dropout
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as par
from sklearn.metrics import precision_score as ps


def custom_loss(y_true, y_pred, w1r, w2f):
    return -K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred) - w1r - w2f)


def enhance_loss(w1r, w2f):
    def loss(y_true, y_pred):
        return custom_loss(y_true, y_pred, w1r, w2f)

    return loss


def test_enhance(y_true, y_pred):
    return -K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))


def returnRiFi():
    fi = pd.read_csv('TestData/InputForEnhanceCNN_label0.csv')
    r1 = fi['Ri'].values
    f1 = fi['Fi'].values
    r1 = r1.tolist()
    f1 = f1.tolist()
    fo = pd.read_csv('TestData/InputForEnhanceCNN_label1.csv')
    r2 = fo['Ri'].values
    f2 = fo['Fi'].values
    for i in r2:
        r1.append(i)
    x = np.asarray(r1)

    for i in f2:
        f1.append(i)
    y = np.asarray(f1)
    return x, y


def returnLabel(count):
    fi = pd.read_csv('TestData/InputForEnhanceCNN_label1.csv')
    x = fi['label']
    x = len(x)
    # arr = np.zeros(count)
    # ind = count-x
    # while (ind < count):
    #     arr[ind] =1
    #     ind+=1

    arr = []
    for i in range(count - x):
        arr.append([0, 1])
    for i in range(x):
        arr.append([1, 0])
    arr = np.asarray(arr)
    return arr


# compute w1,w2
def compute_w1_w2(r, f):
    w1r = []
    w2f = []
    w1 = []
    w2 = []
    w_init = np.random.normal(size=2)
    w1.append(w_init[0])
    w2.append(w_init[1])
    for i in range(1, len(r)):
        w1.append(w1[i - 1] - 0.001 * r[i - 1])
        w2.append(w2[i - 1] - 0.001 * f[i - 1])
    for i in range(0, len(r)):
        w1r.append(w1[i] * r[i])
        w2f.append(w2[i] * f[i])
    return w1r, w2f


def to_matrix(input, shape1, shape2):
    matrix = []
    mtrx = re.sub("\[", ' ', input)
    mtrx = re.sub("]", ' ', mtrx)
    mtrx = mtrx.split()
    count_col = 0
    count_row = 0
    row = ''
    for i in mtrx:
        row = row + i + ' '
        count_col += 1
        if (count_col == shape2) and (count_row < shape1):
            count_col = 0
            count_row += 1
            z = np.fromstring(row, dtype=float, sep=' ')
            # print(z.shape)
            matrix.append(z)
            row = ''

    return matrix


def difLabel(x):
    if (x == [0, 1]):
        return [1, 0]
    else:
        return [0, 1]


def processingLabel(data):
    x = []
    for i in data:
        if i >= 0.5:
            x.append(np.asarray(1))
        else:
            x.append(np.asarray(0))
    return np.asarray(x)

def enhance_CNN():
    # input_vector
    vector = []
    file = pd.read_csv('TestData/InputForEnhanceCNN_label0.csv')
    x = file['input_vector'].values
    count = len(x)
    for i in x:
        mn = to_matrix(i, 2, 300)
        vector.append(mn)

    file = pd.read_csv('TestData/InputForEnhanceCNN_label1.csv')
    x = file['input_vector'].values
    count += len(x)
    for i in x:
        mn = to_matrix(i, 2, 300)
        vector.append(mn)

    # input_vt, x_test, label_train, y_test, rt, rv, ft, fv = train_test_split(vector, label, r,f , test_size=0.25)
    input_vt = np.asarray(vector)
    label = returnLabel(count)
    r, f = returnRiFi()
    w1r, w2f = compute_w1_w2(r, f)
    w1r = np.reshape(w1r, (r.shape[0], 1))
    w2f = np.reshape(w2f, (r.shape[0], 1))
    wr = np.asarray(w1r)
    wf = np.asarray(w2f)

    x, x_test, y, y_test, w1r, w1r_test, w2f, w2f_test = train_test_split(input_vt, label, wr, wf, test_size=0.2)
    x_train, x_val, y_train, y_val, w1r_train, w1r_val, w2f_train, w2f_val = train_test_split(x, y, w1r, w2f,
                                                                                              test_size=0.3)
    
    m, n, p = x_train.shape
    x_train = x_train.reshape(m, n, p, 1)

    m, n, p = x_val.shape
    x_val = x_val.reshape(m, n, p, 1)

    # Build CNN -------------------------------
    model_CNN = Sequential()
    model_CNN.add(Conv2D(100, kernel_size=(2, 2), activation='relu', input_shape=x_train.shape[1:]))
    model_CNN.add(BatchNormalization())
    model_CNN.add(MaxPooling2D(pool_size=(1, 2)))
    model_CNN.add(Dropout(0.5))
    model_CNN.add(Flatten())
    model_CNN.add(Dense(4096, activation='sigmoid'))
    model_CNN.add(Dense(1024, activation='sigmoid'))
    model_CNN.add(Dense(512, activation='sigmoid'))
    model_CNN.add(Dense(256, activation='sigmoid'))
    model_CNN.add(Dense(2, activation='sigmoid'))
    # model_CNN.add(Dropout(0.2))
    model_CNN.compile(optimizer='adam', loss='categorical_crossentropy',  # ,loss=enhance_loss(w1r1, w2f1)
                      metrics=['accuracy'])
    model_CNN.fit(x_train, y_train, epochs=10, batch_size=100, validation_data=(x_val, y_val))
    model_CNN.save("model_enhance.h5")

    # ----------------------------------------------------------------------Test model-------------------------------------------------------------------------------------


    # Test model
    model = load_model('model_enhance.h5')  # custom_objects={''})

    x_test1 = x_test
    y_test1 = y_test
    m, n, p = x_test1.shape
    x_test1 = x_test1.reshape(m, n, p, 1)
    te = 0
    for i in y_test:
        if i == 1:
            te += 1
    result = model.evaluate(x_test1, y_test1, verbose=0)
    print(result)
    result = model.predict(x_test1)

    lab_result = processingLabel(result)
    lab_test = y_test1

    pre, recall, f_beta, support = par(lab_test, lab_result, average='binary')
    print('precision: ', pre)
    print('recall: ', recall)

    print(ps(lab_test, lab_result, average='binary'))
    # print(par(lab_test, lab_result, average='weighted'))


