import gensim
from debias.we import WordEmbedding
from numpy import loadtxt
import os, sys
sys.path.append("..")


def print_similarities(embedding, profession_words):
    # load model as WE class
    embedding_filename =  os.path.join("..", "embeddings", embedding)
    E = WordEmbedding(embedding_filename)

    # load gender direction (from debias function)
    gender_direction = loadtxt(os.path.join("..", "output","gender_direction.csv"), delimiter=',')

    # project professions onto gender dimesion
    sp = sorted([(E.v(w).dot(gender_direction), w) for w in profession_words])

    # print extreme professions
    return sp[:20], sp[-20:]

    # print most similar words to mand/kvinde
    #model.most_similar('ham', topn=10)
    #model.most_similar('hende', topn=10)
