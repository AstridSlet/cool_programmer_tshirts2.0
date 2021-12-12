import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
import pandas as pd
import json
from numpy import loadtxt
if sys.version_info[0] < 3:
    import io
    open = io.open
plt.style.use("seaborn")


model = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin', binary=True) 
debiased_model = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/debiased_model.bin', binary=True)
debiased_model_noeq = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/no_eq_debiased_model.bin', binary=True)

wordlist = ['revisor', 'sygeplejerske','prinsesse', 'dronning','skuespiller', 'skuespillerinde', 'advokat', 'hjælper',  'antropolog', 'arkæolog', 'arkitekt', 'kunstner',  'morder', 'astronaut', 'astronom', 'atlet',  'forfatter', 'bager', 'ballerina', 'fodboldspiller', 'bankmand', 'barber', 'baron', 'bartender', 'biolog', 'biskop', 'præst']
x_ax = ['kvinde', 'mand']
y_ax = np.loadtxt('/work/Exam/dk-weat/output/neutral_specific_difference.csv', delimiter=',')

def plot_professions(embedding, wordlist, x_axis, y_axis):

    wordlist = x_axis + wordlist
    vectors = []

    for i in range(len(wordlist)):
        # Embeddings
        vectors.append(embedding[wordlist[i]])
    # To-be basis
    x = (vectors[1]-vectors[0])
    #flipped
    y = np.flipud(y_axis)
   
    # Get pseudo-inverse matrix
    W = np.array(vectors)
    B = np.array([x,y])
    Bi = np.linalg.pinv(B.T)

    # Project all the words
    Wp = np.matmul(Bi,W.T)
    Wp = (Wp.T-[Wp[0,2],Wp[1,0]]).T
    
    #Wp skal normalizes - virker ikke endnu
    
    Wp = Wp/np.linalg.norm(Wp)#xis=0)
    #Wp = Wp/np.linalg.norm(Wp)
    #PLOT
    plt.figure(figsize=(12,7))
    
    plt.title(label="Professions",
            fontsize=30,
            color="black")
    plt.xlim([-1, 1])
    #plt.ylim([-0.25, 0.25])
    print(Wp[0,0], Wp[1,0])
    
    #plot the wordlist
    plt.scatter(Wp[0,2:], Wp[1,2:], color = 'green', marker= "x")
    #plot kvinde and mand
    plt.scatter(Wp[0,:2], Wp[1,:2], color = 'red', marker= "o", label = "axis of interest")

    rX = max(Wp[0,:])-min(Wp[0,:])
    rX = max(Wp[0,:])-min(Wp[0,:])
    rY = max(Wp[1,:])-min(Wp[1,:])
    eps = 0.005
    
    for i, txt in enumerate(wordlist):
        if txt == "kvinde" or txt == "mand":
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps), fontsize= 'xx-large', c = 'red')
        else:
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps)) # changed from #plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rX*eps))
    plt.axvline(color= 'grey')
    plt.axhline(color= 'grey')
    plt.show()

plot_professions(model, wordlist, x_ax, y_ax)
plot_professions(debiased_model, wordlist, x_ax, y_ax)
plot_professions(debiased_model_noeq, wordlist, x_ax, y_ax)