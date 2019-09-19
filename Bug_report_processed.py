import pandas as pd
from nltk import sent_tokenize,word_tokenize
import numpy as np
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# summary -- word2vec ||| description -- Sent2vec
def word_embedding():
    def tokenize_sentences(corpus):
        return sent_tokenize(corpus)

    def tokenize_corpus(corpus):
        tokens = [x.split() for x in corpus]
        return tokens
    ''''# for Summary
    file = pd.read_csv('Training_Test\AspectJ_process.csv')
    data = str(file['summary'])
    data = re.sub(r'\d', ' ', data)
    corpus = tokenize_sentences(data)
    sentences = tokenize_corpus(corpus)

    # print(sentences)

    # summary -- word2vec ||| description -- Sent2vec
    # word2vec with gensim
    cores = multiprocessing.cpu_count()  # count the number of cores in a computer

    # w2v_model = Word2Vec(tokenized_corpus,size=300, window=2, min_count=1, workers=10, iter=10)
    w2v_model = Word2Vec(min_count=1,
                        window=2,
                        size=300,
                        sample=6e-5,
                        alpha=0.03,
                        min_alpha=0.0007,
                        negative=20,
                        workers=cores - 1,
                        iter=1000)
    w2v_model.build_vocab(sentences, progress_per=10000, update=False)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1, compute_loss=True)
        
    print(w2v_model.wv['bug'])
    print(w2v_model.wv.most_similar(positive=['bug']))
    print(w2v_model.wv.similarity('bug', 'onto')'''

    #for Description

    file = pd.read_csv('Training_Test\AspectJ_process.csv')
    data = file['description'].values
    data = re.sub(r'\d', ' ', str(data))
    
    corpus = sent_tokenize(data)

    sentence = tokenize_corpus(corpus)


    sentences = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentence)]
    # Doc2vec with gensim
    k=300
    model = Doc2Vec(vector_size=k, epochs=40, alpha=0.025, min_alpha=0.00025, min_count = 1, dm=1)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=100)
    model.save('Doc2vec.model')
    del model

