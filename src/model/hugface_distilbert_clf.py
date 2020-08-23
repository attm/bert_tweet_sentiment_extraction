from transformers import TFDistilBertModel


def build_model():
    d_bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    print(type(d_bert_model))

build_model()