import os
from os.path import join as pjoin
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
from src.model.bert_semantic_classifier import get_bert, build_bert_model


cwd = os.getcwd()
BERT_LAYER_HUB_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
BERT_DATASETS_FOLDER_PATH = pjoin(cwd, "data/processed/bert_datasets")
BERT_CHECKPOINT_FOLDER_PATH = pjoin(cwd, "saved_models/bert")

def load_data(datasets_folder : str) -> np.ndarray:
    X_ids = np.load(pjoin(datasets_folder, "X_ids.npy"))
    X_ids_types = np.load(pjoin(datasets_folder, "X_ids_types.npy"))
    X_attn_mask = np.load(pjoin(datasets_folder, "X_attn_mask.npy"))
    y = np.load(pjoin(datasets_folder, "y.npy"))
    return X_ids, X_ids_types, X_attn_mask, y 

def main():
    bert_model = build_bert_model(BERT_LAYER_HUB_URL)
    try:
        bert_model.load_weights(BERT_CHECKPOINT_FOLDER_PATH)
        print(f"\nLoaded weight from {BERT_CHECKPOINT_FOLDER_PATH}")
    except Exception:
        print(f"\nCan't load weight from {BERT_CHECKPOINT_FOLDER_PATH}")

    X_ids, X_ids_types, X_attn_mask, y = load_data(BERT_DATASETS_FOLDER_PATH)
    X = [X_ids, X_ids_types, X_attn_mask]

    cp = tf.keras.callbacks.ModelCheckpoint(BERT_CHECKPOINT_FOLDER_PATH,
                                            save_best_only=True, 
                                            save_weights_only=True)

    bert_model.fit(X, y, 
                   epochs=5,
                   batch_size=32,
                   validation_split=0.2,
                   callbacks=[cp])

if __name__ == "__main__":
    main()
else:
    raise ImportError("Bert_train is the main script, shouldn't be imported.")