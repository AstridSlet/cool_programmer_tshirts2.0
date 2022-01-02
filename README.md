<br />
<p align="center">
  <a href="https://github.com/DaDebias/cool_programmer_tshirts2.0">
    <img src="cool.png" alt="Logo" width=150 height=150>
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

## Usage
This repository contains an example run of debiasing the pre-trained word embedding CONLL-2017 from [daNLP](https://github.com/alexandrainst/danlp). 
To reproduce the analysis, you need to clone this repository and install the required packages with:

```
git clone https://github.com/DaDebias/cool_programmer_tshirts2.0
cd /DaDebias/cool_programmer_tshirts2.0
pip install -r requirements.txt
```
You can then run the pipeline on the CONLL-17 embedding by following the steps below. 

#### Debias 
You first train a classifier to classify if words in the embedding are _gender specific_ or _gender neutral_.

``` 
cd cool_programmer_tshirts2.0/debias
python learn_gender_specific.py --embedding_filename 'conll17.da.wv' --model_alias 'conll17da'
```
Using this list you can now debias the word embedding with: 

```
python debias.py --embedding_filename 'conll17.da.wv' --debiased_filename 'debiased_model.bin' --model_alias 'conll17da'
```
This will produce a debiased version of the word embedding which is saved in the embeddings folder. 

#### Assess bias 
You can now assess bias in the original and debiased word embedding with:

```
python main.py --embedding_filename 'conll17.da.wv' --debiased_filename 'debiased_model.bin' --model_alias 'conll17da'
```

#### Debiasing other word embeddings
The analysis included the application of these two steps on the CONLL-17 model from [daNLP](https://github.com/alexandrainst/danlp). 

If you wish to try the method on other embeddings you simply replace the embedding name in the above line with either of these embedding names: 
- 'conll17.da.wv'
- 'wiki.da.wv'
- 'cc.da.wv'

If you have a downloaded pretrained word embedding as a txt file, you can run the pipeline on this embedding, by placing it in the embeddings folder and running the steps above and replacing the --embedding_filename with the name of the embeddings as well as the --model_alias argument. 

## Contact details
If you have any questions regarding the project itself or the code implementation, feel free to contact us: ([Thea Rolskov Sloth](mailto:201706833@post.au.dk), [Astrid Sletten Rybner](mailto:201808935@post.au.dk))

## Acknowledgements
We would like to give special thanks to the following projects for providing code:
* [Bolukbasi et al. (2016)](https://github.com/tolga-b/debiaswe) authors of the main code used for debiasing word embeddings. 
* [Caliskan et al. (2017)](http://omeka.unibe.ch/files/original/49b5837cb8707025e98129ca035026e0f2143d76.pdf) author of the articles behind the WEAT test.
* [Millie Søndergaard](https://github.com/milsondergaard/speciale) that created the python implementation of the WEAT used for this analysis. 
* [daNLP](https://github.com/alexandrainst/danlp) for providing pre-trained word embeddings. 
* [Sofie Ditmer](https://github.com/sofieditmer) and [Astrid Nørgaard]() for lending us a [word embedding](https://github.com/TheNLPlayPlatform/NLPlay.git) trained on the [Danish Gigaword corpus](https://gigaword.dk/). 
