import os
import sys 
from os.path import join as pjoin
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import bert
import transformers
from transformers import BertTokenizerFast
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
BERT_DATASETS_FOLDER_PATH = pjoin(cwd, "data/processed/bert_datasets")

def main():
    X, y = numpy_datasets_from_csv(TRAIN_CSV_PATH)
    bert_layer, vocab = get_bert(BERT_LAYER_HUB_URL)
    bert_tkn = BertTokenizerFast(vocab)
    X_list = X.tolist()

    X_ids = []
    X_ids_types = []
    X_attn_mask = []

    print(f"Got {len(X)} texts, processing...")
    for txt in X_list:
        ids, types, attention = text_to_bert_ids(bert_tkn, txt)
        X_ids.append(ids)
        X_ids_types.append(types)
        X_attn_mask.append(attention)

    X_ids = np.array(X_ids, dtype=int)
    X_ids_types = np.array(X_ids_types, dtype=int)
    X_attn_mask = np.array(X_attn_mask, dtype=int)

    print("\nGenerated bert datasets:")
    print(f"X_ids shape is {X_ids.shape}")
    print(f"X_ids_types shape is {X_ids_types.shape}")
    print(f"X_attn_mask shape is {X_attn_mask.shape}")
    print(f"y shape is {y.shape}")

    np.save(pjoin(BERT_DATASETS_FOLDER_PATH, "X_ids.npy"), X_ids)
    np.save(pjoin(BERT_DATASETS_FOLDER_PATH, "X_ids_types.npy"), X_ids_types)
    np.save(pjoin(BERT_DATASETS_FOLDER_PATH, "X_attn_mask.npy"), X_attn_mask)
    np.save(pjoin(BERT_DATASETS_FOLDER_PATH, "y.npy"), y)
    print(f"\nSaved datasets to {BERT_DATASETS_FOLDER_PATH}")

if __name__ == "__main__":
    main()
else:
    raise ImportError("Build_bert_dataset is the main script, shouldn't be imported")