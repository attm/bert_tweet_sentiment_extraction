import os
from os.path import join as pjoin
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_hub as tf_hub
from src.model.bert_semantic_classifier import get_bert


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

def build_bert_model(bert_tf_hub_url : str):
    # Getting bert layer
    bert_layer = tf_hub.KerasLayer(BERT_LAYER_HUB_URL, trainable=False)

    # Building model
    ids_input = Input(shape=(128,), dtype=tf.int32, name="input_word_ids")
    ids_tokens_input = Input(shape=(128,), dtype=tf.int32, name="input_type_ids")
    attn_mask = Input(shape=(128,), dtype=tf.int32, name="input_mask")
    bert_input = [ids_input, attn_mask, ids_tokens_input]

    pooled, sequence = bert_layer(bert_input)
    x = Dense(512, activation="relu")(pooled)
    x = Dropout(0.2)(x)
    x = Dense(3, activation="softmax")(x)

    bert_model = Model([ids_input, ids_tokens_input, attn_mask], x)
    bert_model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["acc"])
    return bert_model

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