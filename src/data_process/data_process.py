import os
import numpy as np
import pandas as pd
import transformers
from transformers import BertTokenizerFast


def numpy_datasets_from_csv(csv_path : str, x_column_name : str = "text", y_column_name : str = "sentiment") -> np.ndarray:
    """
    Reads csv file and returns X and y from it. 

    Parameters: 
        csv_path (str) : path of the csv file.
        x_column_name (str) : name of the column with data.
        y_column_name (str) : name of the coulmn with labels. Be advised, labels will be mapped and turned into 0, 1, 2 int's.
    Returns:
        X (np.ndarray) : train data - str's
        y (np.ndarray) : labels - 0, 1, 2
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError("numpy_from_csv: path not exists, given path is {0}".format(csv_path))
    else:
        df = pd.read_csv(csv_path)
        X = df["text"].to_numpy(dtype=str)
        df["sentiment"] = df["sentiment"].map({"negative" : "0", "neutral" : "1", "positive" : "2"})
        y = df["sentiment"].to_numpy(dtype=int)
        return X, y

def text_to_bert(bert_tokenizer : BertTokenizerFast, text : str) -> list:
    """
    Encoding text into bert input.

    Parameters:
        bert_tokenizer (BertTokenizerFast) : bert fast tokenizer form huggingface transformers.
        text (str) : text that needs to be encoded.
    Returns:
        input_ids (list) : tokens ids.
        input_ids_types (list) : types of tokens ids.
        attention_mask (list) : attention mask.
    """
    if not isinstance(bert_tokenizer, BertTokenizerFast):
        raise TypeError(f"text_to_bert: expected bert_tokenizer of type BertTokenizerFast, got {type(bert_tokenizer)}")

    if not isinstance(text, str):
        raise TypeError(f"text_to_bert: expected text of type str, got {type(text)}")

    encoding_dict = bert_tokenizer.encode_plus(text, padding="max_length", max_length=128)

    input_ids = encoding_dict["input_ids"]
    input_ids_types = encoding_dict["token_type_ids"]
    attention_mask = encoding_dict["attention_mask"]

    return input_ids, input_ids_types, attention_mask
