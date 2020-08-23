import numpy as np
import tensorflow_hub as tf_hub


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
    print("\nTrying to load BERT layer from {0}\n".format(tf_hub_url))
    bert_layer = tf_hub.KerasLayer(tf_hub_url, trainable=False)
    print("\nLoaded BERT layer from {0}".format(tf_hub_url))
    # Getting vocab from layer
    vocab_path = bert_layer.resolved_object.vocab_file.asset_path.numpy().decode("utf-8")
    # Creating new tokenizer
    return bert_layer, vocab_path