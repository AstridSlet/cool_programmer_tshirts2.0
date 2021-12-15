from utility_functions import weat_func
from print_similarities import print_similarities
from viz2 import plot_words2
from viz3 import plot_words3
from gensim.models.keyedvectors import KeyedVectors
from danlp.models.embeddings import load_wv_with_gensim
import argparse
import random
import json
import sys, os
sys.path.append(os.path.join('..'))


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_filename", default="DAGW-model(1).bin", help="The name of the embedding. Choose from daNLP: 'conll17.da.wv', 'wiki.da.wv', 'cc.da.wv'")
    parser.add_argument("--debiased_filename", default="debiased_model.bin", help="The name of the embedding")
    parser.add_argument("--model_alias", default = "dagw_word2vec", help="Model alias including embedding type (word2vec, fasttext, wv, etc. or the corpus that it was trained on")

    # parse args
    args = parser.parse_args()
    #k_model_fasttext = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/fasttext_model/fasttext.txt', binary=False)

    print("loading model")
    if args.embedding_filename.endswith(".wv"):
        model = load_wv_with_gensim(args.embedding_filename)
    else:
        model = KeyedVectors.load_word2vec_format(os.path.join("..", "embeddings", args.embedding_filename), binary=True)
     
    print("loading debiased model")
    debiased_model = KeyedVectors.load_word2vec_format(os.path.join("..", "embeddings", f"{args.model_alias}_{args.debiased_filename}"), binary=True)
     
    # define attribute words
    male = ['mandlig', 'mand','dreng','bror','han','ham','hans','søn']
    female = ['kvindelig', 'kvinde', 'pige', 'søster', 'hun', 'hende', 'hendes', 'datter'] 

    # define target words
    science = ['videnskab', 'teknologi', 'fysik', 'kemi', 'computer', 'eksperiment', 'data', 'biologi'] 
    arts = ['poesi', 'kunst', 'dans', 'litteratur', 'roman', 'symfoni', 'drama', 'skulptur'] 
    math = ['matematik', 'algebra', 'geometri', 'regning', 'ligning', 'beregning', 'tal', 'addition'] 
    career = ['leder', 'bestyrelse', 'professionel', 'virksomhed', 'løn', 'arbejde', 'forretning', 'karriere'] 
    family = ['hjem','forældre', 'børn', 'familie','bedsteforældre', 'ægteskab', 'bryllup', 'pårørende'] 
    
    #print("getting WEAT scores")
    #get WEAT scores model
    weat_func(model, f"biased_{args.model_alias}", "career", "family", 10000, male, female, career, family)
    weat_func(model, f"biased_{args.model_alias}", "science", "arts", 10000, male, female, science, arts)
    weat_func(model, f"biased_{args.model_alias}", "math", "arts", 10000, male, female, math, arts)

    # get WEAT scores debiased model
    weat_func(debiased_model, f"debiased_{args.model_alias}", "career", "family", 10000, male, female, career, family)
    weat_func(debiased_model, f"debiased_{args.model_alias}", "science", "arts", 10000, male, female, science, arts)
    weat_func(debiased_model, f"debiased_{args.model_alias}", "math", "arts", 10000, male, female, math, arts)
    
    # load professions
    #professions_path = os.path.join("..", "data", "da_professions.json")
    #with open(professions_path, "r") as f:
        #profession_words = json.load(f)

    # load gender specific words
    #genderspecific_path = os.path.join("..", "data", "da_gender_specific_seed.json")
    #with open(genderspecific_path, "r") as f:
        #gender_specific_words = json.load(f)

    # sample words 
    #profession_sample = random.sample(profession_words, 10)
    #gender_specific_sample = random.sample(gender_specific_words, 10)
    #combined = profession_sample + gender_specific_sample

    # combine wordlist as input to plot and similiarity scores 
    combined = career+family

    # get similarity scores: professions projected onto gender direction
    #print_similarities(args.embedding_filename, combined)
    #print_similarities(args.debiased_filename, combined)
    
    # plot words
    plot_words2(model, f"biased_{args.model_alias}_2", combined)
    plot_words2(debiased_model, f"debiased_{args.model_alias}_2", combined)

    plot_words3(model, f"biased_{args.model_alias}_3", combined)
    plot_words3(debiased_model, f"debiased_{args.model_alias}_3", combined)

    '''
    #hjemmelavet
    Z = ['stærk', 'beslutsom', 'muskler', 'forsørger', 'helt', 'modig', 'kriger', 'stor'] #Target words for Career
    W = ['svag','kærlig', 'diversitet', 'smuk','lille', 'underdanig', 'kreativ', 'hjemmegående'] #Target words for Family

    '''
