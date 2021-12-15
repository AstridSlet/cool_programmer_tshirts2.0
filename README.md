# cool_programmer_tshirts2.0
Code for assessing and removing bias in Danish wordembeddings. 



The method can be used on the following word embeddings:
- 'conll17.da.wv'
- 'wiki.da.wv'
- 'cc.da.wv'

#To run the debiasing method on pre-trained word embeddings from daNLP, the following steps should be followed:

``` 
python learn_gender_specific.py --embedding_filename 'conll17.da.wv' --model_alias 'conll17da'
```
