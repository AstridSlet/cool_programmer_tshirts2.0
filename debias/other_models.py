from __future__ import print_function, division
import re
import os, sys
sys.path.append("..")
import numpy as np
import scipy.sparse
import danlp
#from danlp.models.embeddings import load_wv_with_gensim
import matplotlib.pyplot as plt
plt.style.use("seaborn")
#from danlp.models.embeddings import load_wv_with_gensim
from sklearn.decomposition import PCA
if sys.version_info[0] < 3:
    import io
    open = io.open
else:
    unicode = str

from gensim.models import FastText
model = FastText.load("/work/Exam/cool_programmer_tshirts2.0/embeddings/FT.model")


class WordEmbedding:
    def __init__(self, fname):
        self.thresh = None
        self.max_words = None
        self.desc = fname
        print("*** Reading data from " + fname)
        if fname.endswith(".bin"):
            from gensim.models import KeyedVectors
            model = KeyedVectors.load_word2vec_format(fname, binary=True)
            words = sorted([w for w in model.vocab], key=lambda w: model.vocab[w].index)
            vecs = [model[w] for w in words]
        #elif fname.endswith(".wv"):
            #model = load_wv_with_gensim(fname)
            #words = sorted([w for w in model.vocab], key=lambda w: model.vocab[w].index)
            #vecs = [model[w] for w in words]
        elif fname.endswith(".model"):
            from gensim.models import FastText
            model = FastText.load(fname)
            print("loading words")
            words = sorted([w for w in model.wv], key=lambda w: model.wv[w].index)
            print(words[:5])
            vecs = [model[w] for w in words]
            print(vecs[:5])
        else:
            vecs = []
            words = []

            with open(fname, "r", encoding='utf8') as f:
                for line in f:
                    s = line.split()
                    v = np.array([float(x) for x in s[1:]])
                    if len(vecs) and vecs[-1].shape!=v.shape:
                        print("Got weird line", line)
                        continue
    #                 v /= np.linalg.norm(v)
                    words.append(s[0])
                    vecs.append(v)
        self.vecs = np.array(vecs, dtype='float32')
        print(self.vecs.shape)
        self.words = words
        self.reindex()
        norms = np.linalg.norm(self.vecs, axis=1)
        if max(norms)-min(norms) > 0.0001:
            self.normalize()

    def reindex(self):
        self.index = {w: i for i, w in enumerate(self.words)}
        self.n, self.d = self.vecs.shape
        assert self.n == len(self.words) == len(self.index)
        self._neighbors = None
        print(self.n, "words of dimension", self.d, ":", ", ".join(self.words[:4] + ["..."] + self.words[-4:]))

    def v(self, word):
        return self.vecs[self.index[word]]

    def diff(self, word1, word2):
        v = self.vecs[self.index[word1]] - self.vecs[self.index[word2]]
        return v/np.linalg.norm(v)

    def normalize(self):
        self.desc += ", normalize"
        self.vecs /= np.linalg.norm(self.vecs, axis=1)[:, np.newaxis]
        self.reindex()

    def shrink(self, numwords):
        self.desc += ", shrink " + str(numwords)
        self.filter_words(lambda w: self.index[w]<numwords)

    def filter_words(self, test):
        """
        Keep some words based on test, e.g. lambda x: x.lower()==x
        """
        self.desc += ", filter"
        kept_indices, words = zip(*[[i, w] for i, w in enumerate(self.words) if test(w)])
        self.words = list(words)
        self.vecs = self.vecs[kept_indices, :]
        self.reindex()

    def save(self, filename):
        with open(filename, "w") as f:
            f.write("\n".join([w+" " + " ".join([str(x) for x in v]) for w, v in zip(self.words, self.vecs)]))
        print("Wrote", self.n, "words to", filename)

    def save_w2v(self, filename, binary=True):
        with open(filename, 'wb') as fout:
            fout.write(to_utf8("%s %s\n" % self.vecs.shape))
            # store in sorted order: most frequent words at the top
            for i, word in enumerate(self.words):
                row = self.vecs[i]
                if binary:
                    fout.write(to_utf8(word) + b" " + row.tostring())
                else:
                    fout.write(to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))

    def remove_directions(self, directions): #directions better be orthogonal
        self.desc += ", removed"
        for direction in directions:
            self.desc += " "
            if type(direction) is np.ndarray:
                v = direction / np.linalg.norm(direction)
                self.desc += "vector "
            else:
                w1, w2 = direction
                v = self.diff(w1, w2)
                self.desc += w1 + "-" + w2
            self.vecs = self.vecs - self.vecs.dot(v)[:, np.newaxis].dot(v[np.newaxis, :])
        self.normalize()

    def compute_neighbors_if_necessary(self, thresh, max_words):
        thresh = float(thresh) # dang python 2.7!
        if self._neighbors is not None and self.thresh == thresh and self.max_words == max_words:
            return
        print("Computing neighbors")
        self.thresh = thresh
        self.max_words = max_words
        vecs = self.vecs[:max_words]
        dots = vecs.dot(vecs.T)
        dots = scipy.sparse.csr_matrix(dots * (dots >= 1-thresh/2))
        from collections import Counter
        rows, cols = dots.nonzero()
        nums = list(Counter(rows).values())
        print("Mean:", np.mean(nums)-1)
        print("Median:", np.median(nums)-1)
        rows, cols, vecs = zip(*[(i, j, vecs[i]-vecs[j]) for i, j, x in zip(rows, cols, dots.data) if i<j])
        self._neighbors = rows, cols, np.array([v/np.linalg.norm(v) for v in vecs])

    def neighbors(self, word, thresh=1):
        dots = self.vecs.dot(self.v(word))
        return [self.words[i] for i, dot in enumerate(dots) if dot >= 1-thresh/2]

    def more_words_like_these(self, words, topn=50, max_freq=100000):
        v = sum(self.v(w) for w in words)
        dots = self.vecs[:max_freq].dot(v)
        thresh = sorted(dots)[-topn]
        words = [w for w, dot in zip(self.words, dots) if dot>=thresh]
        return sorted(words, key=lambda w: self.v(w).dot(v))[-topn:][::-1]

    def best_analogies_dist_thresh(self, v, thresh=1, topn=500, max_words=50000):
        """Metric is cos(a-c, b-d) if |b-d|^2 < thresh, otherwise 0
        """
        vecs, vocab = self.vecs[:max_words], self.words[:max_words]
        self.compute_neighbors_if_necessary(thresh, max_words)
        rows, cols, vecs = self._neighbors
        scores = vecs.dot(v/np.linalg.norm(v))
        pi = np.argsort(-abs(scores))

        ans = []
        usedL = set()
        usedR = set()
        for i in pi:
            if abs(scores[i])<0.001:
                break
            row = rows[i] if scores[i] > 0 else cols[i]
            col = cols[i] if scores[i] > 0 else rows[i]
            if row in usedL or col in usedR:
                continue
            usedL.add(row)
            usedR.add(col)
            ans.append((vocab[row], vocab[col], abs(scores[i])))
            if len(ans)==topn:
                break

        return ans

def doPCA(pairs, embedding, num_components = 0.95):
    matrix = []
    for a, b in pairs:
        center = (embedding.v(a) + embedding.v(b))/2
        matrix.append(embedding.v(a) - center)
        matrix.append(embedding.v(b) - center)
    matrix = np.array(matrix)
    pca = PCA(n_components = num_components)
    pca.fit(matrix)
    return pca

def plotPCA(pca, model_type, n_components):
    plt.bar(range(pca.n_components_), pca.explained_variance_ratio_, color = "seagreen")
    plt.title(f"Explained variance by PCA components - model: {model_type}")
    plt.xlabel("PCA components")
    plt.ylabel("Explained variance")
    print ("PCA plot saved to output folder")
    plt.savefig(os.path.join("..", "output", f"{model_type}_pca_plot.png"))

    
def drop(u, v):
    return u - v * u.dot(v) / v.dot(v)
