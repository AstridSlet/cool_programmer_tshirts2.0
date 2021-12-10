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
if sys.version_info[0] < 3:
    import io
    open = io.open


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_filename", type=str, default="DAGW-model(1).bin", help="Filename embedding")
    parser.add_argument("--NUM_TRAINING", type=int, default = 50000, help="N words in training set")
    parser.add_argument("--GENDER_SPECIFIC_SEED_WORDS", type=str, default="da_gender_specific_seed.json", help="Filename gender specific seed")
    parser.add_argument("--outfile", type=str, default = "gender_specific_full.json", help="Filename gender specific full")

    # parse arguments
    args = parser.parse_args()

    # retrieve args
    embedding_filename =  os.path.join("..", "embeddings", args.embedding_filename)
    NUM_TRAINING = args.NUM_TRAINING
    GENDER_SPECIFIC_SEED_WORDS = os.path.join("..", "data", args.GENDER_SPECIFIC_SEED_WORDS)
    OUTFILE = os.path.join("..", "data", args.outfile)



    print("Loading embedding...")
    E = WordEmbedding(embedding_filename)

    print("Embedding has {} words.".format(len(E.words)))
    print("{} seed words from '{}' out of which {} are in the embedding.".format(
        len(gender_seed),
        GENDER_SPECIFIC_SEED_WORDS,
        len([w for w in gender_seed if w in E.words]))
)

    gender_seed = set(w for i, w in enumerate(E.words) if w in gender_seed or (w.lower() in gender_seed and i<NUM_TRAINING))
    labeled_train = [(i, 1 if w in gender_seed else 0) for i, w in enumerate(E.words) if (i<NUM_TRAINING or w in gender_seed)]

    # assign index number and labels to seperate variables
    train_indices, train_labels = zip(*labeled_train)

    # define train and test as arrays
    y = np.array(train_labels)
    X = np.array([E.vecs[i] for i in train_indices])

    # define penalization
    C = 1.0
    clf = LinearSVC(C=C, tol=0.0001)
    clf.fit(X, y)
    weights = (0.5 / (sum(y)) * y + 0.5 / (sum(1 - y)) * (1 - y))
    weights = 1.0 / len(y)
    score = sum((clf.predict(X) == y) * weights)
    print(1 - score, sum(y) * 1.0 / len(y))

    pred = clf.coef_[0].dot(X.T)
    direction = clf.coef_[0]
    intercept = clf.intercept_

    is_gender_specific = (E.vecs.dot(clf.coef_.T) > -clf.intercept_)

    full_gender_specific = list(set([w for label, w in zip(is_gender_specific, E.words)
                                if label]).union(gender_seed))
    full_gender_specific.sort(key=lambda w: E.index[w])


    # save gender direction 
    savetxt(os.path.join("..", "output", "neutral_specific_difference.csv"), direction, delimiter=',')

    with open(outfile, "w") as f:
        json.dump(full_gender_specific, f)
        