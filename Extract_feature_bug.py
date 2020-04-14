import numpy as np
from keras.models import Model,Sequential
from keras.layers import Conv1D, MaxPooling1D,Input,Embedding
import  Bug_word_embedding as bwe
import pickle

def ExtractVector(data, maxSent):
    with open("Bug_extract_feature.txt", "wb") as f:
        vectors = []
        filterSizes = [2, 3, 4]
        for matrix in data:
            result = []
            batch = np.expand_dims(matrix, axis=0)
            for filterSize in filterSizes:
                model_CNN = Sequential()
                model_CNN.add(Conv1D(100, filterSize, activation='relu', name='conv1d', input_shape=matrix.shape))
                model_CNN.add(MaxPooling1D(maxSent - filterSize + 1))
                model_CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                conv_vector = model_CNN.predict(batch)
                result.append(conv_vector.reshape(1, 100))
            result = np.asarray(result)
            result = result.reshape(1, 300)
            vectors.append(result)
        vectors = np.asarray(vectors)
        pickle.dump(vectors,f)
        return vectors
