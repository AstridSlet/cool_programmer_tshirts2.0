import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
import pandas as pd
import json
from numpy import loadtxt
import sys, os
sys.path.append(os.path.join('..'))
if sys.version_info[0] < 3:
    import io
    open = io.open
plt.style.use("seaborn")
 

model = KeyedVectors.load_word2vec_format('/work/Exam/cool_programmer_tshirts2.0/embeddings/conll17da_debiased_model.bin', binary=True) 


def equalize_visualization(embedding, eq_pairs, gn_word, model_alias, plot_title):

    # load x-axis
    y = np.loadtxt(os.path.join("..", "output", f"{model_alias}_gender_subspace.csv"), delimiter=',')

    #combine
    wordlist = gn_word + eq_pairs
    
    # choose only words that are in the embeddings
    wordlist = [w for w in wordlist if w in embedding.vocab]
    
    # retrieve vectors
    vectors = [embedding[k] for k in wordlist]

  
    x = vectors[0]
    
    # Get pseudo-inverse matrix
    W = np.array(vectors)
    B = np.array([x,y])
    Bi = np.linalg.pinv(B.T)

    # Project all the words
    Wp = np.matmul(Bi,W.T)
    Wp = (Wp.T-[Wp[0,2],Wp[1,0]]).T
    
    #PLOT
    plt.figure(figsize=(12,7))
    
    plt.title(label=plot_title,
            fontsize=30,
            color="black")
    #plt.xlim([-0.8, 0.8])
    #plt.ylim([-0.25, 0.25])
    plt.axvline(color= 'lightgrey')
    plt.axhline(color= 'lightgrey')
    
    plt.xlabel("Projection of 'school'")
    plt.ylabel("Gender subspace")
    #plot the wordlist
    plt.scatter(Wp[0,:1],Wp[1,:1], color = 'black', marker= "x", s = 200)
    plt.scatter(Wp[0,1:3],Wp[1,1:3], color = 'mediumseagreen', marker= "o")
    plt.plot(Wp[0,1:3],Wp[1,1:3], color = 'mediumseagreen', marker= "o", alpha =0.3, linestyle='dashed')
    plt.scatter(Wp[0,3:5],Wp[1,3:5], color = 'darkorchid', marker= "o")
    plt.plot(Wp[0,3:5],Wp[1,3:5], color = 'darkorchid', marker= "o", alpha =0.3, linestyle='dashed')
    plt.scatter(Wp[0,5:7],Wp[1,5:7], color = 'dodgerblue', marker= "o")
    plt.plot(Wp[0,5:7],Wp[1,5:7], color = 'dodgerblue', marker= "o", alpha =0.3, linestyle='dashed')
    plt.scatter(Wp[0,7:9],Wp[1,7:9], color = 'indianred', marker= "o")
    plt.plot(Wp[0,7:9],Wp[1,7:9], color = 'indianred', marker= "o", alpha =0.3, linestyle='dashed')
    
    plt.plot()
    rX = max(Wp[0,:])-min(Wp[0,:])
    rY = max(Wp[1,:])-min(Wp[1,:])
    eps = 0.005
    
    for i, txt in enumerate(wordlist):
        if txt == "skole":
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps), fontsize= 'xx-large', c = 'black')
        else:
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps), fontsize = 'large') # changed from #plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rX*eps))
    
    plt.savefig(os.path.join("..", "output", f"{model_alias}_eq_plot.png"))

eq_pairs = ['mand', 'kvinde', 'pige', 'dreng', 'bedstemor', 'bedstefar', 'mor', 'far']
gn_word = ["skole"]
model_alias = "conll17da"
plot_title = "Equalized Pairs Projected on the Gender Neutral Word 'School'"
equalize_visualization(model, eq_pairs, gn_word, model_alias, plot_title)
