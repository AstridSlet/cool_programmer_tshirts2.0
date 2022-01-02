<br />
<p align="center">
  <a href="https://github.com/DaDebias/cool_programmer_tshirts2.0">
    <img src="readme_images/cool.png" alt="Logo" width=150 height=150>
  </a>
  
  <h1 align="center">Exam Project: Natural Language Processing</h1> 
  <h3 align="center">MSc Cognitive Science 2021</h3> 


  <p align="center">
    Thea Rolskov Sloth & Astrid Sletten Rybner
    <br />
    <a 
    Aarhus University
    a>
    <br />
  </p>
</p>


## Project information
This repository contains code for reproducing our analysis regarding gender bias in Danish pre-trained word embeddings. The steps include: 1) removing gender bias in the word embeddings with hard-debiasing [Bolukbasi et al. (2016)](https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf), and 2) Assessing bias in the word embeddings with the Word Embeddings Association Test (WEAT) [Caliskan et al. (2017)](http://omeka.unibe.ch/files/original/49b5837cb8707025e98129ca035026e0f2143d76.pdf).

The first part (removing bias) produces a debiased version of the input word-embedding, which is saved to the folder 'embeddings'. 

The second part (assessing bias) produces WEAT scores for two of the gender biases from Caliskan et al. (2017): career-family and math-arts. 

## Repository structure
The repository includes one folder containing the code for each of these two steps. The output from running each step is saved to the output folder. 

| Folder | Description|
|--------|:-----------|
```assess_bias``` | scripts for assessing bias 
```debias``` | scripts for removing bias 
```embeddings```| folder for original/debiased embeddings
```output``` | output folder for output WEAT scores and plots
```readme_images``` | images used in the readme files

## Usage

The original and debiased embeddings are placed in the embeddings folder.  

The analysis included the application of these two steps on 

The method can be used on the following pre-trained word embeddings from [daNLP](https://github.com/alexandrainst/danlp):
- 'conll17.da.wv'
- 'wiki.da.wv'
- 'cc.da.wv'


#### To run the debiasing method on pre-trained word embeddings from daNLP, the following steps should be followed:

##### Train a classifier to classify if a word is _gender specific_ or _gender neutral_.
``` 
python learn_gender_specific.py --embedding_filename 'conll17.da.wv' --model_alias 'conll17da'
```

##### Debias the word embedding:

```
python debias.py --embedding_filename 'conll17.da.wv' --debiased_filename 'debiased_model.bin' --model_alias 'conll17da'
```

##### Assess the original embedding and its debiased version:
```
python main.py --embedding_filename 'conll17.da.wv' --debiased_filename 'debiased_model.bin' --model_alias 'conll17da'
```
