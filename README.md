## Deep Learning Semantic Analysis using BERT

Simple learning project. 
Goal is to predict semantic label for tweets. Dataset is from https://www.kaggle.com/c/tweet-sentiment-extraction

### Requirments
Developed in docker. Used tensorflow-gpu official image with additional libraries installed, check requirments.txt

### Data preparation
Tokenized and lemmatized with NLTK
Encoded with huggingface transformers BertTokenizerFast

### Models
1. bert-base from tf hub, with softmax layer on top
2. distilBert from huggingface transformers