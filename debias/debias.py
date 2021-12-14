from __future__ import print_function, division
import we
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys, os
sys.path.append("..")
if sys.version_info[0] < 3:
    import io
    open = io.open
"""
Hard-debias embedding

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""


def debias(E, gender_specific_words, definitional, equalize, model_type):
    # do PCA analysis
    pca = we.doPCA(definitional, E, num_components=10)
    
    # plot PCA
    we.plotPCA(pca, model_type, n_components=0.95)

    # get gender direction as csv file
    gender_direction = pca.components_[0]

    specific_set = set(gender_specific_words)
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    E.normalize()
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    print(candidates)
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            y = we.drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_filename", default="DAGW-model(1).bin", help="The name of the embedding. Choose from daNLP: 'conll17.da.wv', 'wiki.da.wv', 'cc.da.wv'")
    parser.add_argument("--definitional_filename", default = "da_definitional_pairs.json", help="Definitional pairs for creating gender direction")
    parser.add_argument("--gendered_words_filename", default = "da_gender_specific_full.json",help="File containing words not to neutralize (one per line)")
    parser.add_argument("--equalize_filename", default = "da_equalize_pairs.json", help="Word pairs for equalizing")
    parser.add_argument("--debiased_filename", default = "debiased_model.bin", help="???.bin")
    parser.add_argument("--model_alias", default = "dagw_word2vec", help="Model alias including embedding type (word2vec, fasttext, wv, etc. or the corpus that it was trained on")

    # parse args
    args = parser.parse_args()

    # retrieve args
    if args.embedding_filename.endswith(".wv"):
        embedding_filename = args.embedding_filename
    else:
        embedding_filename = os.path.join("..","embeddings", args.embedding_filename)
    definitional_filename = os.path.join("..","data", args.definitional_filename)
    print(definitional_filename)
    gendered_words_filename = os.path.join("..","data", args.gendered_words_filename)
    equalize_filename = os.path.join("..","data", args.equalize_filename)
    model_alias = args.model_alias
    debiased_filename = os.path.join("..","embeddings", f"{model_alias}_{args.debiased_filename}")
    
    with open(definitional_filename, "r") as f:
        defs = json.load(f)
    print("definitional", defs)

    with open(equalize_filename, "r") as f:
        equalize_pairs = json.load(f)

    with open(gendered_words_filename, "r") as f:
        gender_specific_words = json.load(f)
    print("gender specific", len(gender_specific_words), gender_specific_words[:10])

    
    # load embeddings
    E = we.WordEmbedding(embedding_filename)

    print("Debiasing:", embedding_filename)
    debias(E, gender_specific_words, defs, equalize_pairs, model_alias)

    print("Saving to file...")
    if embedding_filename[-4:] == debiased_filename[-4:] == ".bin":
        E.save_w2v(debiased_filename)
        print("I saved with save_w2v-function")
    else:
        E.save_w2v(debiased_filename)
        print("I saved here with save-function") #where wv files save at least
    print("\n\nDone!\n")