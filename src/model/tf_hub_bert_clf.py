import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_hub as tf_hub


BERT_LAYER_HUB_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"


def get_bert(tf_hub_url : str) -> np.ndarray:
    """
    Downloads bert layer.

    Parameters:
        tf_hub_url (str) : url of the bert model.
    Returns:
        bert_layer (str) : keras bert layer.
        vocab_path (path) : vocab file path used for tokenizing.
    """
    # Loading bert from tf hub
    print(f"\nTrying to load BERT layer from {tf_hub_url}\n")
    bert_layer = tf_hub.KerasLayer(tf_hub_url, trainable=True)
    print(f"\nLoaded BERT layer from {tf_hub_url}")
    # Getting vocab from layer
    vocab_path = bert_layer.resolved_object.vocab_file.asset_path.numpy().decode("utf-8")
    # Creating new tokenizer
    return bert_layer, vocab_path

def build_bert_model(bert_tf_hub_url : str = BERT_LAYER_HUB_URL):
    """
    Builds bert based model. 

    Parameters:
        bert_tf_hub_url (str) : url of the bert model from the tf hub.
    Returns:
        bert_model (tf.keras.Model) : built model.
    """
    # Getting bert layer
    bert_layer = tf_hub.KerasLayer(bert_tf_hub_url, trainable=False)

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
    bert_model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return bert_model
