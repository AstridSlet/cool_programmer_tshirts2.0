import random
from danlp.models.embeddings import load_wv_with_gensim
import json
import os
import pandas as pd

print("running")
# load embedding
embedding = load_wv_with_gensim('conll17.da.wv')

# get keys
words = sorted([w for w in embedding.vocab])

# create filter to remove words with numbers
filter = [any(str.isdigit(c) for c in word) for word in words]

# get words using filters
filtered_list = [i for indx,i in enumerate(words) if filter[indx] == False]

# sample 2000 words
sample = random.sample(filtered_list, 2000)

# define outfile
outfile = os.path.join("..", "data", "gender_neutral.csv")

# save file
with open(outfile, "w", encoding='utf8') as f:
    json.dump(sample, f, ensure_ascii=False)
