import gensim
from debias.we import WordEmbedding
from numpy import loadtxt
import os, sys
sys.path.append("..")
from gensim.models.keyedvectors import KeyedVectors


def print_similarities(embedding, words_list):
    # load model as WE class
    embedding_filename =  os.path.join("..", "embeddings", embedding)
    E = WordEmbedding(embedding_filename)

    # load gender direction (from debias function)
    #gender_direction = loadtxt(os.path.join("..", "output","gender_direction.csv"), delimiter=',')

    gender_direction =  E.v("kvinde") - E.v("mand")
    print("These are the values in kvinde-mand")
    print(gender_direction)

    # project professions onto gender dimesion
    sp = sorted([(E.v(w).dot(gender_direction), w) for w in words_list])

    # print extreme professions
    return print(f"Top five words:{sp[:5]} Bottom five words: {sp[-5:]}")


'''
# print most similar words to mand/kvinde
model.most_similar('ham', topn=10)
model.most_similar('hende', topn=10)

def analogy(word_a, word_b, word_c, embedding):
    result = embedding.most_similar(negative=[word_a], positive=[word_b, word_c], topn=15)
    return result

model = KeyedVectors.load_word2vec_format('/work/Exam/cool_programmer_tshirts2.0/embeddings/DAGW-model(1).bin', binary=True) 
debiased_model = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/debiased_model.bin', binary=True)
analogy('mand', 'læge', 'kvinde',debiased_model)
analogy('mand', 'læge', 'kvinde', model)
analogy('mand', 'programmør', 'kvinde',debiased_model)
analogy('mand', 'programmør', 'kvinde', model)
'''