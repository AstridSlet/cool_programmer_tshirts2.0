

def debias(E, gender_specific_words, definitional, equalize):
    
    # do PCA on definitional word pairs
    num_components=10
    pca = we.doPCA(definitional, E, num_components)
    plt.bar(range(num_components), pca.explained_variance_ratio_)
    plt.savefig(os.path.join("..", "output", "pca_plot.png"))

    # use top component as gender direction
    gender_direction = pca.components_[0]
    
    #gender_direction = we.plotPCA(E, definitional, 0.95)
    
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

