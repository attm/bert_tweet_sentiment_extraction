from transformers import TFDistilBertForSequenceClassification as TFBert
from transformers import DistilBertTokenizerFast
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def get_tokenizer() -> DistilBertTokenizerFast:
    """
    Returns tokenizer for that model.

    Parameters:
        None
    Returns:
        tokenizer (DistilBertTokenizerFast) : loaded and set tokenizer.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    return tokenizer

def build_model():
    """
    Builds model based on huggingface DistilBert.

    Parameters:
        None
    Returns:
        model (tf.keras.Model) : compiled model, ready to train.
    """
    d_bert_model = TFBert.from_pretrained('distilbert-base-uncased')
    # Building model
    ids_input = Input(shape=(128,), dtype=tf.int32, name="input_word_ids")
    attn_mask = Input(shape=(128,), dtype=tf.int32, name="input_mask")

    pooled = d_bert_model([ids_input, attn_mask])[0]
    x = Dense(64, activation="relu")(pooled)
    x = Dropout(0.2)(x)
    x = Dense(3, activation="softmax")(x)

    bert_model = Model([ids_input, attn_mask], x)
    bert_model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return bert_model
