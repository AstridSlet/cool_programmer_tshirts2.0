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


def plot_words(embedding, model_name, wordlist_full):

    # load x-axis
    x = np.loadtxt(os.path.join("..", "output","gender_direction.csv"), delimiter=',')

    # load vector for y-axis
    y_ax = np.loadtxt(os.path.join("..", "output", "neutral_specific_difference.csv"), delimiter=',')

    # choose only words that are in the embeddings
    wordlist = [w for w in wordlist_full if w in embedding.vocab]

    # retrieve vectors
    vectors = [embedding[k] for k in wordlist]

    # To-be basis
    #x = (vectors[1]-vectors[0])
    
    # flipped
    y = np.flipud(y_ax)

    # normalize
    #x /= np.linalg.norm(x)
    #y /= np.linalg.norm(y)
   
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
    plt.xlim([-0.75, 0.75])
    #plt.ylim([-0.25, 0.25])

    
    #plot the wordlist
    plt.scatter(Wp[0,2:], Wp[1,2:], color = 'green', marker= "x")
    #plot kvinde and mand
    #plt.scatter(Wp[0,:2], Wp[1,:2], color = 'red', marker= "o", label = "axis of interest")

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
    plt.savefig(os.path.join("..", "output", f"{model_name}_gender_plot.png"))
