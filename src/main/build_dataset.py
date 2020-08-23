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
from src.data_process.dataset import build_bert_datasets
from src.model.tf_hub_bert_clf import get_bert


cwd = os.getcwd()
# Data folders
RAW_DATA_FOLDER_PATH = pjoin(cwd, "data/raw")
TRAIN_CSV_PATH = pjoin(RAW_DATA_FOLDER_PATH, "train.csv")
TEST_CSV_PATH = pjoin(RAW_DATA_FOLDER_PATH, "test.csv")
DATASETS_FOLDER_PATH = pjoin(cwd, "data/datasets")
TF_HUB_BERT_FOLDER_PATH = pjoin(DATASETS_FOLDER_PATH, "tf_hub_bert")
# TF_HUB BERT
TF_HUB_BERT_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"

def build_tf_hub_bert_datasets(datasets_save_folder_path : str) -> None:
    # Generating and saving train
    bert_layer, bert_vocab = get_bert(TF_HUB_BERT_URL)
    bert_tokenizer = BertTokenizerFast(bert_vocab)
    X_train_ids, X_train_ids_types, X_train_attn_mask, y_train = build_bert_datasets(TRAIN_CSV_PATH, bert_tokenizer)

    np.save(pjoin(datasets_save_folder_path, "X_train_ids.npy"), X_train_ids)
    np.save(pjoin(datasets_save_folder_path, "X_train_ids_types.npy"), X_train_ids_types)
    np.save(pjoin(datasets_save_folder_path, "X_train_attn_mask.npy"), X_train_attn_mask)
    np.save(pjoin(datasets_save_folder_path, "y_train.npy"), y_train)
    print(f"\nSaved train datasets to {datasets_save_folder_path}")

    # Generating and saving test
    X_test_ids, X_test_ids_types, X_test_attn_mask, y_test = build_bert_datasets(TEST_CSV_PATH, bert_tokenizer)

    np.save(pjoin(datasets_save_folder_path, "X_test_ids.npy"), X_test_ids)
    np.save(pjoin(datasets_save_folder_path, "X_test_ids_types.npy"), X_test_ids_types)
    np.save(pjoin(datasets_save_folder_path, "X_test_attn_mask.npy"), X_test_attn_mask)
    print(f"\nSaved train datasets to {datasets_save_folder_path}")

def main():
    # Building tf_hub bert datasets
    build_tf_hub_bert_datasets(TF_HUB_BERT_FOLDER_PATH)

if __name__ == "__main__":
    main()
else:
    raise ImportError("Build_bert_dataset is the main script, shouldn't be imported")