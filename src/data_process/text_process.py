import bert


def text_to_bert_ids(bert_tokenizer : bert.bert_tokenization.FullTokenizer, text : str) -> list:
    if not isinstance(bert_tokenizer, bert.bert_tokenization.FullTokenizer):
        raise TypeError("text_to_bert_ids: expected bert_tokenizer of type bert.bert_tokenization.FullTokenizer, got {0}".format(type(bert_tokenizer)))

    if not isinstance(text, str):
        raise TypeError("text_to_bert_ids: expected text of type str, got {0}".format(type(text)))

    ids = bert_tokenizer.tokenize(text)
    ids = bert_tokenizer.convert_tokens_to_ids(ids)
    return ids

def bert_ids_to_text(bert_tokenizer : bert.bert_tokenization.FullTokenizer, ids : list) -> str:
    if not isinstance(bert_tokenizer, bert.bert_tokenization.FullTokenizer):
        raise TypeError("bert_ids_to_text: expected bert_tokenizer of type bert.bert_tokenization.FullTokenizer, got {0}".format(type(bert_tokenizer)))

    tokens = bert_tokenizer.convert_ids_to_tokens(ids)
    text = " ".join(tokens)
    return text