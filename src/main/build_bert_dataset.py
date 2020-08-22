import os
import sys 
from os.path import join as pjoin
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import bert
from src.data_process.dataset_builder import numpy_datasets_from_csv
from src.data_process.text_process import text_to_bert_ids, bert_ids_to_text
from src.model.bert_semantic_classifier import get_bert


cwd = os.getcwd()
# Text data path's
RAW_DATA_FOLDER_PATH = pjoin(cwd, "data/raw")
TRAIN_CSV_PATH = pjoin(RAW_DATA_FOLDER_PATH, "train.csv")
TEST_CSV_PATH = pjoin(RAW_DATA_FOLDER_PATH, "test.csv")
# Bert data
BERT_LAYER_HUB_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
BERT_SAVED_DATA_FOLDER_PATH = pjoin(cwd, "data/bert_data")

def main():
    X, y = numpy_datasets_from_csv(TRAIN_CSV_PATH)
    bert_layer, bert_tokenizer = get_bert(BERT_LAYER_HUB_URL)
    X_list = X.tolist()

    

if __name__ == "__main__":
    main()
else:
    raise ImportError("Build_bert_dataset is the main script, shouldn't be imported")