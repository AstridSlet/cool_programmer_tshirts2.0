import gensim
import json
from debias.we import WordEmbedding
from numpy import loadtxt
import os, sys
sys.path.append("..")


def print_similarities(embedding):
    # load model as WE class
    embedding_filename =  os.path.join("..", "embeddings", embedding)
    E = WordEmbedding(embedding_filename)
    
    # load professions
    professions_path = os.path.join("..", "data", "professions.json")
    with open(professions_path, "r") as f:
        profession_words = json.load(f)

    # load gender direction (from debias function)
    gender_direction = loadtxt(os.path.join("..", "output","gender_direction.csv"), delimiter=',')

    # project professions onto gender dimesion
    sp = sorted([(E.v(w).dot(gender_direction), w) for w in profession_words])

    # print extreme professions
    return sp[:20], sp[-20:]

    # print most similar words to mand/kvinde
    #model.most_similar('ham', topn=10)
    #model.most_similar('hende', topn=10)
