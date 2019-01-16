# DLDIY Project: NLP style transfer

The goal in this project is to train a sentence embedding model using various sources in order to be able to project sentences in a specific author corpus.

## Datasets

The used data is a combination of Kaggle datasets, collections from the Gutenberg project and the Brown dataset.
Gutenberg and brown datasets can be automatically gathered from the Internet using the `gather.py` script. 
Kaggle datasets must be manually downloaded in order to be preprocessed.

We'll try to do transfer on:
* Authors: (Shakespeare, Dickens, Wilde, Austin)
* Songs
* Hillary Clinton leaked e-mails

## Method

The skip-thought vector encoding [(paper)](https://arxiv.org/abs/1506.06726) will be used to learn a representation of the natural language. After that, we should be able given a sentence 
to find the closest sentence written by a given author. 

## Results

Coming soon.