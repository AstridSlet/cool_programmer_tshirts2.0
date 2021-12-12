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


def debias(E, gender_specific_words, definitional, equalize):
    # do PCA on definitional word pairs
    num_components=10
    pca = we.doPCA(definitional, E, num_components)
    plt.bar(range(num_components), pca.explained_variance_ratio_)
    plt.savefig(os.path.join("..", "output", "pca_plot.png"))

    # use top component as gender direction
    gender_direction = pca.components_[0]

    # save gender direction (to print most extreme job professions)
    np.savetxt(os.path.join("..", "output", "gender_direction.csv"), gender_direction, delimiter=',')

    # load full genderspecific
    specific_set = set(gender_specific_words)

    # neutralize: go through entire wordembedding - remove  gender direction from words not in full gender specific
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    
    E.normalize()

    # equalize: take all equalize pairs (both in upper/lowercanse) 
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
    parser.add_argument("--embedding_filename", default="DAGW-model(1).bin", help="The name of the embedding")
    parser.add_argument("--definitional_filename", default = "da_definitional_pairs.json", help="Definitional pairs for creating gender direction")
    parser.add_argument("--gendered_words_filename", default = "da_gender_specific_full.json",help="File containing words not to neutralize (one per line)")
    parser.add_argument("--equalize_filename", default = "da_equalize_pairs.json", help="Word pairs for equalizing")
    parser.add_argument("--debiased_filename", default = "debiased_model.bin", help="???.bin")

    # parse args
    args = parser.parse_args()

    # retrieve args
    embedding_filename = os.path.join("..","embeddings", args.embedding_filename)
    definitional_filename = os.path.join("..","data", args.definitional_filename)
    print(definitional_filename)
    gendered_words_filename = os.path.join("..","data", args.gendered_words_filename)
    equalize_filename = os.path.join("..","data", args.equalize_filename)
    debiased_filename = os.path.join("..","embeddings", args.debiased_filename)


    with open(definitional_filename, "r") as f:
        defs = json.load(f)
    print("definitional", defs)

    with open(equalize_filename, "r") as f:
        equalize_pairs = json.load(f)

    with open(gendered_words_filename, "r") as f:
        gender_specific_words = json.load(f)
    print("gender specific", len(gender_specific_words), gender_specific_words[:10])

    E = we.WordEmbedding(embedding_filename)

    print("Debiasing...")
    debias(E, gender_specific_words, defs, equalize_pairs)

    print("Saving to file...")
    if embedding_filename[-4:] == debiased_filename[-4:] == ".bin":
        E.save_w2v(debiased_filename)
    else:
        E.save(debiased_filename)

    print("\n\nDone!\n")
