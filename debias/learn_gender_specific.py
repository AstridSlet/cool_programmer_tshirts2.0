"""
Learn gender specific words

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""
from __future__ import print_function, division
import os, sys
sys.path.append("..")
import argparse
from we import *
from sklearn.svm import LinearSVC
import json
from numpy import savetxt
if sys.version_info[0] < 3:
    import io
    open = io.open


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_filename", default="DAGW-model(1).bin", help="The name of the embedding. Choose from daNLP: 'conll17.da.wv', 'wiki.da.wv', 'cc.da.wv'")    
    parser.add_argument("--num_training", type=int, default = 50000, help="N words in training set")
    parser.add_argument("--gender_specific_seed_words", type=str, default="da_gender_specific_seed.json", help="Filename gender specific seed")
    parser.add_argument("--outfile", type=str, default = "da_gender_specific_full.json", help="Filename gender specific full")
    parser.add_argument("--model_alias", default = "dagw_word2vec", help="Model alias including embedding type (word2vec, fasttext, wv, etc. or the corpus that it was trained on")

    # parse arguments
    args = parser.parse_args()

    # load from embeddings folder or danlo 
    if args.embedding_filename.endswith(".wv"):
        embedding_filename = args.embedding_filename
    else:
        embedding_filename = os.path.join("..","embeddings", args.embedding_filename)
    
    # retrieve args 
    num_training = args.num_training
    gender_specific_seed_words = os.path.join("..", "data", args.gender_specific_seed_words)
    outfile = os.path.join("..", "data", args.outfile)

    # open seed words file
    with open(gender_specific_seed_words, "r") as f:
        gender_seed = json.load(f)

    print("Loading embedding...")
    E = WordEmbedding(embedding_filename)

    print("Embedding has {} words.".format(len(E.words)))
    print("{} seed words from '{}' out of which {} are in the embedding.".format(
        len(gender_seed),
        gender_specific_seed_words,
        len([w for w in gender_seed if w in E.words]))
)
    # retrieve vectors for gender specific words to train on
    gender_seed = set(w for i, w in enumerate(E.words) if w in gender_seed or (w.lower() in gender_seed and i<num_training))
    # label vectors (1: gender specific, 0: non gender specific)
    labeled_train = [(i, 1 if w in gender_seed else 0) for i, w in enumerate(E.words) if (i<num_training or w in gender_seed)]

    # assign index number and labels to seperate variables
    train_indices, train_labels = zip(*labeled_train)

    # define train and test as arrays
    y = np.array(train_labels)
    X = np.array([E.vecs[i] for i in train_indices])

    # define penalization
    C = 1.0

    # define model 
    clf = LinearSVC(C=C, tol=0.0001)
    
    # fit model 
    clf.fit(X, y)
    weights = (0.5 / (sum(y)) * y + 0.5 / (sum(1 - y)) * (1 - y))
    weights = 1.0 / len(y)
    score = sum((clf.predict(X) == y) * weights)
    print(1 - score, sum(y) * 1.0 / len(y))

    # make predictions for words in training set 
    pred = clf.coef_[0].dot(X.T)
    direction = clf.coef_[0]
    intercept = clf.intercept_

    # make predictions for words NOT in training set 
    is_gender_specific = (E.vecs.dot(clf.coef_.T) > -clf.intercept_)

    # combine seed words data with the rest of the gender_specific words 
    full_gender_specific = list(set([w for label, w in zip(is_gender_specific, E.words)
                                if label]).union(gender_seed))
    
    # full gender specific data
    full_gender_specific.sort(key=lambda w: E.index[w])


    # save gender neutrality vector 
    savetxt(os.path.join("..", "output", f"{args.model_alias}_neutrality.csv"), direction, delimiter=',')

    # save full gender specific
    with open(outfile, "w") as f:
        json.dump(full_gender_specific, f)
    
    print("\n\nDone!\n")
        