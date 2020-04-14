from gensim.models import Word2Vec
import multiprocessing as mul
import numpy as np
import time
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Embedding
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


# vector of line
def Sourcefile_CombineVector(line_sent, tf_idf, model):
    vector_line = np.zeros(shape=(1, 300))
    count = 0
    for word_sent in line_sent:
        vector_line += model.wv[word_sent] * tf_idf['tfidf'][
            word_sent]  # cần truyền vào tf_idf ở trên def, k truyền thì nó hiểu làm sao tf_idf là gì?
        count += 1
    vector_line = vector_line / count
    return vector_line


def mutliprocessing_sourcefile(corpus, model, tf_idf):
    # compute line max
    line_max = 0
    for ptu in corpus:
        if (len(ptu) > line_max):
            line_max = len(ptu)
    t1 = time.time()
    # mutliprocess
    number_processing = 2
    group_size = 500
    groups = []
    for i in range(number_processing):
        if i == number_processing - 1:
            groups.append(corpus[i * group_size:])
        else:
            groups.append(corpus[i * group_size: (i + 1) * group_size])
    p = mul.Pool(processes=number_processing)
    multi_res = [p.apply_async(Sourcefile_Word_embedding, (group, tf_idf, line_max, model)) for group in groups]
    res = [res.get() for res in multi_res]
    t2 = time.time()
    print(t2 - t1)
    return res


# wordembedding sourcefile
def Sourcefile_Word_embedding(group, tf_idf, line_max, model):
    matrix = []
    for sourcefile in group:
        vecto = np.zeros(shape=(line_max, 300))
        for i in range(0, len(sourcefile)):
            vecto[i] = Sourcefile_CombineVector(sourcefile[i], tf_idf, model)
        matrix.append(vecto)
    return matrix


# extract vector with CNN
def feature_detect(matrix):
    vectors = []
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


def sourcefile_extractvector(srs, tf_idf):
    model = Word2Vec.load("w2v_model.model")
    res = mutliprocessing_sourcefile(srs, model, tf_idf)
    matrixs = []
    for i in res:
        for j in i:
            matrixs.append(j)
    vectors = feature_detect(matrixs)
    return vectors

