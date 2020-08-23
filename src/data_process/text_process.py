import transformers
from transformers import BertTokenizerFast


def text_to_bert_ids(bert_tokenizer : BertTokenizerFast, text : str) -> list:
    """
    Encoding text into bert input.

    Parameters:
        bert_tokenizer (BertTokenizerFast) : bert fast tokenizer form huggingface transformers.
        text (str) : text that needs to be encoded.
    Returns:
        input_ids (list) : tokens ids.
        token_type_ids (list) : types of tokens ids.
        attention_mask (list) : attention mask.
    """
    if not isinstance(bert_tokenizer, BertTokenizerFast):
        raise TypeError("text_to_bert_ids: expected bert_tokenizer of type BertTokenizerFast, got {0}".format(type(bert_tokenizer)))

    if not isinstance(text, str):
        raise TypeError("text_to_bert_ids: expected text of type str, got {0}".format(type(text)))

    encoding_dict = bert_tokenizer.encode_plus(text, padding="max_length", max_length=128)

    input_ids = encoding_dict["input_ids"]
    token_type_ids = encoding_dict["token_type_ids"]
    attention_mask = encoding_dict["attention_mask"]

    return input_ids, token_type_ids, attention_mask

def bert_ids_to_text(bert_tokenizer : BertTokenizerFast, ids : list) -> str:
    if not isinstance(bert_tokenizer, BertTokenizerFast):
        raise TypeError("bert_ids_to_text: expected bert_tokenizer of type BertTokenizerFast, got {0}".format(type(bert_tokenizer)))

    tokens = bert_tokenizer.convert_ids_to_tokens(ids)
    text = " ".join(tokens)
    return text