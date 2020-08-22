import numpy as np
import tensorflow_hub as tf_hub
import bert


def get_bert(tf_hub_url : str) -> np.ndarray:
    """
    Downloads bert layer and creates new bert tokenizer from tensorflow hub.

    Parameters:
        tf_hub_url (str) : url of the bert model.
    Returns:
        bert_layer (str) :
        bert_tokenizer (str) :
    """
    # Loading bert from tf hub
    print("\nTrying to load BERT layer from {0}\n".format(tf_hub_url))
    bert_layer = tf_hub.KerasLayer(tf_hub_url, trainable=False)
    print("\nLoaded BERT layer from {0}".format(tf_hub_url))
    # Getting vocab from layer
    vocab = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    # Creating new tokenizer
    bert_tokenizer = bert.bert_tokenization.FullTokenizer(vocab, do_lower_case=True)
    print("Bert layer and tokenizer successfully loaded.\n")
    return bert_layer, bert_tokenizer