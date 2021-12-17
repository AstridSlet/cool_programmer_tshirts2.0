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
from sklearn.manifold import TSNE

def plot_words(embedding, model_alias, professions, gender_specific, bias_type, biased):

    # load x-axis
    x = np.loadtxt(os.path.join("..", "output", f"{model_alias}_gender_subspace.csv"), delimiter=',')

    #x_ax = ['mand', 'kvinde']

    # load vector for y-axis
    y_ax = np.loadtxt(os.path.join("..", "output", f"{model_alias}_neutrality.csv"), delimiter=',')

    #combine
    wordlist = professions+gender_specific
    
    # choose only words that are in the embeddings
    wordlist = [w for w in wordlist if w in embedding.vocab]
    
    # retrieve vectors
    vectors = [embedding[k] for k in wordlist]

    # To-be basis
    #x = (vectors[1]-vectors[0])
    
    # flipped
    y = np.flipud(y_ax)
    
    # normalize
    x/= np.linalg.norm(x)
    y/= np.linalg.norm(y)
        
    # Get pseudo-inverse matrix
    W = np.array(vectors)
    B = np.array([x,y])
    Bi = np.linalg.pinv(B.T)

    # Project all the words
    Wp = np.matmul(Bi,W.T)
    Wp = (Wp.T-[Wp[0,2],Wp[1,0]]).T
      
    #Wp = Wp/np.linalg.norm(Wp)#xis=0)
    #Wp = Wp/np.linalg.norm(Wp)
    
    #PLOT
    plt.figure(figsize=(12,7))
    
    plt.title(label=bias_type,
            fontsize=30,
            color="black")
    #plt.xlim([-0.8, 0.8])
    #plt.ylim([-0.25, 0.25])
    plt.axvline(color= 'lightgrey')
    plt.axhline(color= 'lightgrey')
    plt.xlabel("Gender Subspace")
    plt.ylabel("Gender Neutrality")
    
    #plot the wordlist
    plt.scatter(Wp[0,:int(len(professions))], 
        Wp[1,:int(len(professions))], 
        color = 'orchid', marker= "o")
    plt.scatter(Wp[0,int(len(professions)):int(len(professions+gender_specific))],
        Wp[1,int(len(professions)):int(len(professions+gender_specific))], 
        color = 'mediumseagreen', marker= "o")
    #plt.scatter(Wp[0,:2], Wp[1,:2], color = 'lightgrey', marker= "x")

    rX = max(Wp[0,:])-min(Wp[0,:])
    rY = max(Wp[1,:])-min(Wp[1,:])
    eps = 0.005
    
    for i, txt in enumerate(wordlist):
        if txt == "kvinde" or txt == "mand":
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps), fontsize= 'xx-large', c = 'black')
        else:
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps)) # changed from #plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rX*eps))
    
    plt.savefig(os.path.join("..", "output", f"{biased}_{model_alias}_2_gender_plot.png"))

def restrict_wv(wv, restricted_word_set):
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []

    for i in range(len(wv.vocab)):
        word = wv.index2entity[i]
        vec = wv.vectors[i]
        vocab = wv.vocab[word]
        vec_norm = wv.vectors_norm[i]
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
            new_vectors_norm.append(vec_norm)

    wv.vocab = new_vocab
    wv.vectors = np.array(new_vectors)
    wv.index2entity = np.array(new_index2entity)
    wv.index2word = np.array(new_index2entity)
    wv.vectors_norm = np.array(new_vectors_norm)
 
def tsne_plot(model, model_alias):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(n_components=2, perplexity =10)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(12, 12), dpi=600) 
    for i in range(len(x)):
        plt.title(label=f"TSNE plot: {model_alias}",
            fontsize=30,
            color="black")
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(os.path.join("..", "output", f"tsne_plot{model_alias}.png"))
    plt.show()

