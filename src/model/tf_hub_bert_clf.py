import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_hub as tf_hub
from transformers import BertTokenizerFast


BERT_LAYER_HUB_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"


def get_tokenizer() -> BertTokenizerFast:
    """
    Returns tokenizer for that model.

    Parameters:
        None
    Returns:
        tokenizer (BertTokenizerFast) : loaded and set tokenizer.
    """
    # Loading bert from tf hub
    print(f"\nTrying to load BERT layer from {BERT_LAYER_HUB_URL}\n")
    bert_layer = tf_hub.KerasLayer(BERT_LAYER_HUB_URL, trainable=False)
    print(f"\nLoaded BERT layer from {BERT_LAYER_HUB_URL}")
    # Getting vocab from layer
    vocab_path = bert_layer.resolved_object.vocab_file.asset_path.numpy().decode("utf-8")
    # Creating new tokenizer
    tokenizer = BertTokenizerFast(vocab_path)
    return tokenizer

def build_model(bert_tf_hub_url : str = BERT_LAYER_HUB_URL):
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
    x = Dense(3, activation="softmax")(pooled)

    bert_model = Model([ids_input, ids_tokens_input, attn_mask], x)
    bert_model.compile(optimizer=Adam(learning_rate=0.01), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return bert_model
