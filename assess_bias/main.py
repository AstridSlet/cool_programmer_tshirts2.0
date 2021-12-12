from utility_functions import *
from print_similarities import * 
from gensim.models.keyedvectors import KeyedVectors
import argparse
import sys, os
sys.path.append(os.path.join('..'))
#from viz import *



if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_filename", default="DAGW-model(1).bin", help="The name of the embedding")
    parser.add_argument("--debiased_filename", default="debiased_model.bin", help="The name of the embedding")
    parser.add_argument("--model_type", default = "word2vec", help="Model type e.g. word2vec, fasttext etc.")

    # parse args
    args = parser.parse_args()

    #k_model_fasttext = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/fasttext_model/fasttext.txt', binary=False)

    print("loading model")
    model = KeyedVectors.load_word2vec_format(os.path.join("..", "embeddings", args.embedding_filename), binary=True) 
    
    print("loading debiased model")
    debiased_model = KeyedVectors.load_word2vec_format(os.path.join("..", "embeddings", args.debiased_filename), binary=True)

    # define attribute words
    male = ['mandlig', 'mand','dreng','bror','han','ham','hans','søn'] 
    female = ['kvindelig', 'kvinde', 'pige', 'søster', 'hun', 'hende', 'hendes', 'datter'] 

    # define target words
    science = ['videnskab', 'teknologi', 'fysik', 'kemi', 'computer', 'eksperiment', 'data', 'biologi', 'mand'] 
    arts = ['poesi', 'kunst', 'dans', 'litteratur', 'roman', 'symfoni', 'drama', 'skulptur', 'kvinde'] 
    math = ['matematik', 'algebra', 'geometri', 'regning', 'ligning', 'beregning', 'tal', 'addition'] 
    career = ['leder', 'bestyrelse', 'professionel', 'virksomhed', 'løn', 'arbejde', 'forretning', 'karriere'] 
    family = ['hjem','forældre', 'børn', 'familie','bedsteforældre', 'ægteskab', 'bryllup', 'pårørende'] 

    # get WEAT scores model
    #weat_func(model, args.model_type, "career", "family", 10000, male, female, career, family)
    #weat_func(model, args.model_type, "science", "arts", 10000, male, female, science, arts)
    #weat_func(model, args.model_type, "math", "arts", 10000, male, female, math, arts)

    # get WEAT scores debiased model
    #weat_func(debiased_model, args.model_type, "career", "family", 10000, male, female, career, family)
    #weat_func(debiased_model, args.model_type, "science", "arts", 10000, male, female, science, arts)
    #weat_func(debiased_model, args.model_type, "math", "arts", 10000, male, female, math, arts)

    print_similarities(model)
    print_similarities(debiased_model)


    #words_to_plot = set(male)
    # restrict embedding
    #restrict_wv(wv = model, restricted_word_set = words_to_plot)
    # plot
    #tsne_plot(model)

    '''
    #hjemmelavet
    Z = ['stærk', 'beslutsom', 'muskler', 'forsørger', 'helt', 'modig', 'kriger', 'stor'] #Target words for Career
    W = ['svag','kærlig', 'diversitet', 'smuk','lille', 'underdanig', 'kreativ', 'hjemmegående'] #Target words for Family

    '''
