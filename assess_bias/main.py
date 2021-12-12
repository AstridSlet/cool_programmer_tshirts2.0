
from utility_functions import *
from viz import restrict_wv
from gensim.models.keyedvectors import KeyedVectors
import sys, os
sys.path.append(os.path.join('..'))

# define name 

model_name = "word2vec"

# load model 
#k_model_fasttext = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/fasttext_model/fasttext.txt', binary=False)
model = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin', binary=True) 
debiased_model = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/debiased_model.bin', binary=True)

male = ['mandlig', 'mand','dreng','bror','han','ham','hans','søn'] #Attribute words for mand
female = ['kvindelig', 'kvinde', 'pige', 'søster', 'hun', 'hende', 'hendes', 'datter'] #Attribute words for kvinde

science = ['videnskab', 'teknologi', 'fysik', 'kemi', 'computer', 'eksperiment', 'data', 'biologi', 'mand'] #Target words for Science
arts = ['poesi', 'kunst', 'dans', 'litteratur', 'roman', 'symfoni', 'drama', 'skulptur', 'kvinde'] #Target words for Arts
math = ['matematik', 'algebra', 'geometri', 'regning', 'ligning', 'beregning', 'tal', 'addition'] #Target words for Math
career = ['leder', 'bestyrelse', 'professionel', 'virksomhed', 'løn', 'arbejde', 'forretning', 'karriere'] #Target words for Career
family = ['hjem','forældre', 'børn', 'familie','bedsteforældre', 'ægteskab', 'bryllup', 'pårørende'] #Target words for Family

weat_func(model, model_name, "career", "family", 10000, male, female, career, family)
weat_func(model, model_name, "science", "arts", 10000, male, female, science, arts)
weat_func(model, model_name, "math", "arts", 10000, male, female, math, arts)

weat_func(debiased_model, model_name, "career", "family", 10000, male, female, career, family)
weat_func(debiased_model, model_name, "science", "arts", 10000, male, female, science, arts)
weat_func(debiased_model, model_name, "math", "arts", 10000, male, female, math, arts)



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
