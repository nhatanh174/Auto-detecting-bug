import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D,Flatten,Input,Embedding
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

def feature_detect (matrix):
    vectors=[]
    for source in matrix:
        input = np.expand_dims(source, axis=0)
        vector_feature = np.zeros((3, 100))
        num = 0
        for i in range(2, 5):
            model = Sequential()
            model.add(Conv1D(activation='relu', filters=100, kernel_size=i, input_shape=source.shape))
            model.add(MaxPooling1D(pool_size=source.shape[0] - i + 1))
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
            vector = np.reshape(model.predict(input), (1, 100))
            vector_feature[num] = vector
            num += 1

        vectors.append(vector_feature.flatten())
    return vectors