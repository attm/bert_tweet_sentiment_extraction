import os
from os.path import join as pjoin
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
from src.model.tf_hub_bert_clf import build_model


cwd = os.getcwd()
BERT_DATASETS_FOLDER_PATH = pjoin(cwd, "data/datasets/tf_hub_bert")
TF_HUB_BERT_CP_PATH = pjoin(cwd, "saved_models/tf_hub_bert/model")
DISTIL_BERT_CP_PATH = pjoin(cwd, "saved_models/distil_bert/model")

def load_data(datasets_folder : str) -> np.ndarray:
    X_train_ids = np.load(pjoin(datasets_folder, "X_train_ids.npy"))
    X_train_ids_types = np.load(pjoin(datasets_folder, "X_train_ids_types.npy"))
    X_train_attn_mask = np.load(pjoin(datasets_folder, "X_train_attn_mask.npy"))
    y_train = np.load(pjoin(datasets_folder, "y_train.npy"))
    return X_train_ids, X_train_ids_types, X_train_attn_mask, y_train 

def train_tf_hub_bert_clf(cp_path : str):
    tf.keras.backend.clear_session()
    model = build_model()
    print(model.summary())
    try:
        model.load_weights(cp_path)
        print(f"\nLoaded weight from {cp_path}")
    except Exception:
        print(f"\nCan't load weight from {cp_path}")

    X_ids, X_ids_types, X_attn_mask, y = load_data(BERT_DATASETS_FOLDER_PATH)
    X = [X_ids, X_ids_types, X_attn_mask]

    cp = tf.keras.callbacks.ModelCheckpoint(cp_path,
                                            save_best_only=True, 
                                            save_weights_only=True)

    model.fit(X, y, 
                   epochs=5,
                   batch_size=4,
                   validation_split=0.2,
                   callbacks=[cp])

def main():
    train_tf_hub_bert_clf(TF_HUB_BERT_CP_PATH)

if __name__ == "__main__":
    main()
else:
    raise ImportError("Bert_train is the main script, shouldn't be imported.")