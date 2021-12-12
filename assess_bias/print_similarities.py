from we import *
import gensim
import json
from numpy import loadtxt


def print_similarities(model):
    # load professions
    professions_path = os.path.join("..", "data", "professions.json")
    with open(professions_path, "r") as f:
        profession_words = json.load(f)

    # load gender direction (from debias function)
    direction = loadtxt(os.path.join("..", "output","gender_direction.csv", delimiter=',')

    # load models 
    #embedding_filename =  ("/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin")
    #E = WordEmbedding(embedding_filename)
    #model = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin', binary=True) 

    # print most similar words to mand/kvinde
    model.most_similar('ham', topn=10)
    model.most_similar('hende', topn=10)

    # project professions onto gender dimesion
    sp = sorted([(E.v(w).dot(direction), w) for w in profession_words])

    # print extreme professions
    sp[:20], sp[-20:]

