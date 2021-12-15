# cool_programmer_tshirts2.0
Code for assessing and removing bias in Danish wordembeddings. 



The method can be used on the following word embeddings:
- 'conll17.da.wv'
- 'wiki.da.wv'
- 'cc.da.wv'

# To run the debiasing method on pre-trained word embeddings from daNLP, the following steps should be followed:

## Train a classifier to classify if a word is _gender specific_ or _gender neutral_.
``` 
python learn_gender_specific.py --embedding_filename 'conll17.da.wv' --model_alias 'conll17da'
```

## Debias the word embedding:

```
python debias.py --embedding_filename 'conll17.da.wv' --debiased_filename 'debiased_model.bin' --model_alias 'conll17da'
```
