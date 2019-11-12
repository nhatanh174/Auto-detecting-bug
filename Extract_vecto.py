import numpy as np
import TF_IDF
from gensim.models import Word2Vec

# vector of line
def CombineVector(line_sent,df):
    model = Word2Vec.load('w2v_model.model')
    vector_line = np.zeros(shape=(1,300))
    count=0
    for word_sent in  line_sent:
        vector_line += model.wv[word_sent]*df['tfidf'][word_sent]
        count+=1
    vector_line=vector_line/count
    return vector_line
