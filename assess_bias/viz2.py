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
 

def plot_words2(embedding, model_alias, professions, gender_specific, bias_type, biased):

    # load x-axis
    #x = np.loadtxt(os.path.join("..", "output", f"{model_alias}_gender_subspace.csv"), delimiter=',')

    x_ax = ['mand', 'kvinde']

    # load vector for y-axis
    y_ax = np.loadtxt(os.path.join("..", "output", f"{model_alias}_neutrality.csv"), delimiter=',')

    #combine
    wordlist = x_ax+professions+gender_specific
    
    # choose only words that are in the embeddings
    wordlist = [w for w in wordlist if w in embedding.vocab]
    
    # retrieve vectors
    vectors = [embedding[k] for k in wordlist]

    # To-be basis
    x = (vectors[1]-vectors[0])
    
    # flipped
    y = np.flipud(y_ax)

    # normalize
    #x /= np.linalg.norm(x)
    y/= np.linalg.norm(y)
   
    # Get pseudo-inverse matrix
    W = np.array(vectors)
    B = np.array([x,y])
    Bi = np.linalg.pinv(B.T)

    # Project all the words
    Wp = np.matmul(Bi,W.T)
    Wp = (Wp.T-[Wp[0,2],Wp[1,0]]).T
    
    #Wp skal normalizes - virker ikke endnu
    
    #Wp = Wp/np.linalg.norm(Wp)#xis=0)
    #Wp = Wp/np.linalg.norm(Wp)
    
    #PLOT
    plt.figure(figsize=(12,7))
    
    plt.title(label=bias_type,
            fontsize=30,
            color="black")
    plt.xlim([-0.8, 0.8])
    #plt.ylim([-0.25, 0.25])

    #plot the wordlist
    plt.scatter(Wp[0,int(len(x_ax)):int(len(professions))], Wp[1,int(len(x_ax)):int(len(professions))], color = 'orchid', marker= "o")
    plt.scatter(Wp[0,int(len(professions)+int(len(x_ax))):int(len(professions+gender_specific)+int(len(x_ax)))], Wp[1,int(len(professions)+int(len(x_ax))):int(len(professions+gender_specific)+int(len(x_ax)))], color = 'mediumseagreen', marker= "o")
    plt.scatter(Wp[0,:int(len(x_ax))], Wp[1,:int(len(x_ax))], color = 'black', marker= "x")

    rX = max(Wp[0,:])-min(Wp[0,:])
    #rX = max(Wp[0,:])-min(Wp[0,:])
    rY = max(Wp[1,:])-min(Wp[1,:])
    eps = 0.005
    
    for i, txt in enumerate(wordlist):
        if txt == "kvinde" or txt == "mand":
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps), fontsize= 'xx-large', c = 'black')
        else:
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps)) # changed from #plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rX*eps))
    plt.axvline(color= 'lightgrey')
    plt.axhline(color= 'lightgrey')
    plt.savefig(os.path.join("..", "output", f"{biased}_{model_alias}_2_gender_plot.png"))
