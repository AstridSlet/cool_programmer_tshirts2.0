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



'''
def plot_words3(embedding, model_alias, wordlist_input, bias_type, biased):

    # load x-axis
    x_ax = ['mand', 'kvinde'] #np.loadtxt(os.path.join("..", "output","gender_direction.csv"), delimiter=',')

    # load vector for y-axis
    y_ax = ['familie', 'karriere'] #np.loadtxt(os.path.join("..", "output", "neutral_specific_difference.csv"), delimiter=',')

    # combine wordlist
    wordlist_full = x_ax + y_ax + wordlist_input

    # choose only words that are in the embeddings
    wordlist = [w for w in wordlist_full if w in embedding.vocab]
    
    # retrieve vectors
    vectors = [embedding[k] for k in wordlist]

    # To-be basis
    x = (vectors[1]-vectors[0])
    
    # flipped
    y = np.flipud(vectors[3]-vectors[2])

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
    
    plt.title(label=bias_type,
            fontsize=30,
            color="black")
    plt.xlim([-0.75, 0.75])
    #plt.ylim([-0.25, 0.25])

    
    #plot the wordlist
    plt.scatter(Wp[0,2:], Wp[1,2:], color = 'green', marker= "x")
    #plot kvinde and mand
    plt.scatter(Wp[0,:2], Wp[1,:2], color = 'red', marker= "o", label = "axis of interest")

    rX = max(Wp[0,:])-min(Wp[0,:])
    #rX = max(Wp[0,:])-min(Wp[0,:])
    rY = max(Wp[1,:])-min(Wp[1,:])
    eps = 0.005
    
    for i, txt in enumerate(wordlist):
        if txt == "kvinde" or txt == "mand":
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps), fontsize= 'xx-large', c = 'red')
        else:
            plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps)) # changed from #plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rX*eps))
    plt.axvline(color= 'grey')
    plt.axhline(color= 'grey')
    plt.savefig(os.path.join("..", "output", f"{biased}{model_alias}_3gender_plot.png"))





model_name = "test"
model = KeyedVectors.load_word2vec_format('/work/Exam/cool_programmer_tshirts2.0/embeddings/DAGW-model(1).bin', binary=True) 

Wl = ['kvinde', 'mand','familie', 'karriere', 'dronning',' konge','pige','leder', 'bestyrelse', 'professionel', 'virksomhed', 'løn', 'arbejde', 'forretning', 'hjem','forældre', 'børn','bedsteforældre', 'ægteskab', 'bryllup', 'pårørende'] 

Wv = []
for i in range(len(Wl)):
    Wv.append(model[Wl[i]])
b1 = (Wv[1]-Wv[0])
b2 = (Wv[3]-Wv[2])

W = np.array(Wv)
B = np.array([b1,b2])
Bi = np.linalg.pinv(B.T)

Wp = np.matmul(Bi,W.T)
print(Wp.shape)
Wp = (Wp.T-[Wp[0,2],Wp[1,0]]).T

plt.figure(figsize=(12,7))
plt.axvline()
plt.axhline()
plt.scatter(Wp[0,:], Wp[1,:])
rX = max(Wp[0,:])-min(Wp[0,:])
rY = max(Wp[1,:])-min(Wp[1,:])
eps = 0.005
for i, txt in enumerate(Wl):
    plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rX*eps))
plt.show()

'''

'''

Wl = ['skole', 'mand', 'kvinde', 'pige', 'dreng', 'bedstemor', 'bedstefar', 'mor', 'far'] 

Wv = []
for i in range(len(Wl)):
    Wv.append(model[Wl[i]])
b1 = Wv[0]
b2 = (Wv[2]-Wv[1])

W = np.array(Wv)
B = np.array([b1,b2])
Bi = np.linalg.pinv(B.T)

Wp = np.matmul(Bi,W.T)
print(Wp.shape)
Wp = (Wp.T-[Wp[0,2],Wp[1,0]]).T

plt.figure(figsize=(12,7))
plt.axvline()
plt.axhline()
plt.scatter(Wp[0,:], Wp[1,:])
rX = max(Wp[0,:])-min(Wp[0,:])
rY = max(Wp[1,:])-min(Wp[1,:])
eps = 0.005
for i, txt in enumerate(Wl):
    plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rX*eps))
plt.show()
'''