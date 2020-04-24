import multiprocessing
from gensim.models import Word2Vec
import pandas as pd
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import sent_tokenize

# Define constance
BUG="TestData\Eclipse_Platform_UI.txt"
BUG_CSV ="TestData\Eclipse_Platform_UI_csv.csv"
BUG_PROCESS = "TestData\Eclipse_Platform_UI_process.csv"

# buil model word2vec for summary in bug reports
def splitData(data):
    x=[]
    for line in data:
        x.append(line.split())
    return x

def bugWord2vec(data):
    cores = multiprocessing.cpu_count()  # count the number of cores in a computer
    w2vModel = Word2Vec(min_count=1,
                        window=2,
                        size=300,
                        sample=6e-5,
                        alpha=0.03,
                        min_alpha=0.0007,
                        negative=20,
                        workers=cores - 1,
                        iter=10)
    w2vModel.build_vocab(data, progress_per=10000)
    w2vModel.train(data, total_examples=w2vModel.corpus_count, epochs=30, report_delay=1, compute_loss=True)
    w2vModel.save('w2v_summary.model')
    del w2vModel

# build model doc2vec for description
def desDoc2vec():
    def tokenize_corpus(corpus):
        tokens = [x.split() for x in corpus]
        return tokens
    file = pd.read_csv(BUG_PROCESS)
    data = file['description'].values
    data = re.sub(r'\d', ' ', str(data))
    corpus = sent_tokenize(data)
    sentence = tokenize_corpus(corpus)
    sentences = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentence)]
    # Doc2vec with gensim
    k = 300
    model = Doc2Vec(vector_size=k, epochs=40, alpha=0.025, min_alpha=0.00025, min_count=1, dm=1)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=100)
    model.save('Doc2vec.model')
    del model
file = pd.read_csv(BUG_PROCESS)
data = file['summary'].values
data = splitData(data)
bugWord2vec(data)
desDoc2vec()
