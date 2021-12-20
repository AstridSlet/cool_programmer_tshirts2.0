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

    # load vector for y-axis
    y_ax = np.loadtxt(os.path.join("..", "output", f"{model_alias}_neutrality.csv"), delimiter=',')

    #combine
    wordlist = professions+gender_specific
    
    # choose only words that are in the embeddings
    wordlist = [w for w in wordlist if w in embedding.vocab]
    
    # retrieve vectors
    vectors = [embedding[k] for k in wordlist]
   
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
    plt.xlabel("Gender Subspace", fontsize = 20)
    plt.ylabel("Gender Neutrality", fontsize = 20)
    
    #plot the wordlist
    gender_specific_scatter = plt.scatter(Wp[0,:int(len(professions))], 
        Wp[1,:int(len(professions))], 
        color = 'darkorchid', marker= "o")
    professions_scatter = plt.scatter(Wp[0,int(len(professions)):int(len(professions+gender_specific))],
        Wp[1,int(len(professions)):int(len(professions+gender_specific))], 
        color = 'mediumseagreen', marker= "o")
    #plt.scatter(Wp[0,:2], Wp[1,:2], color = 'lightgrey', marker= "x")
    plt.legend((gender_specific_scatter, professions_scatter),('Professions', 'Gender specific'),scatterpoints=1,loc='upper left',ncol=1,fontsize=15, facecolor='white', framealpha=1, frameon=True)

    rX = max(Wp[0,:])-min(Wp[0,:])
    rY = max(Wp[1,:])-min(Wp[1,:])
    eps = 0.005
    
    for i, txt in enumerate(wordlist):
        if txt == "kvinde" or txt == "mand":
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps), fontsize= 'xx-large', c = 'black')
        else:
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps), fontsize = 'large') # changed from #plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rX*eps))
    
    plt.savefig(os.path.join("..", "output", f"{biased}_{model_alias}_gender_plot.png"))

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
 
def tsne_plot(model, model_alias, male, female, career, family, math, arts):
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
    plt.title(label=f"t-SNE plot: {model_alias}",
            fontsize=30,
            color="black")
    # add dots
    for label,i in zip(labels, range(len(x))):
        if label in male:
            a = plt.scatter(x[i],y[i], color="b")
        elif label in female:
            b = plt.scatter(x[i],y[i], color="c")
        elif label in career:
            c = plt.scatter(x[i],y[i], color="r")
        elif label in family:
            d = plt.scatter(x[i],y[i], color="g")
        elif label in math:
            e = plt.scatter(x[i],y[i], color="m")
        elif label in arts:
            f = plt.scatter(x[i],y[i], color="y")
    # add labels
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     fontsize = 14,
                     ha='right',
                     va='bottom')
    plt.legend((a, b, c, d, e, f),("male", "female", "career", "family", "math", "arts"),scatterpoints=1,ncol=2,fontsize=15, facecolor='white', framealpha=1, frameon=True)
    plt.savefig(os.path.join("..", "output", f"tsne_plot_{model_alias}.png"))
    plt.show()


def equalize_visualization(embedding, eq_pairs, gn_word, model_alias, biased, plot_title):

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
    
    plt.xlabel("Projection of school", fontsize=22)
    plt.ylabel("Gender subspace", fontsize=22)
    
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
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*(eps+0.02)), fontsize= 'xx-large', c = 'black')
        else:
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps), fontsize = 'large') # changed from #plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rX*eps))
    
    plt.savefig(os.path.join("..", "output", f"{biased}{model_alias}_eq_plot.png"))
