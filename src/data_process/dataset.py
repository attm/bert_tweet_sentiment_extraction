import numpy as np
from transformers import BertTokenizer
from src.data_process.data_process import numpy_datasets_from_csv, text_to_bert


def build_bert_datasets(csv_path : str, tokenizer : BertTokenizer) -> np.ndarray:
    X, y = numpy_datasets_from_csv(csv_path)
    X_list = X.tolist()

    X_ids = []
    X_ids_types = []
    X_attn_mask = []

    print(f"Got {len(X)} texts, processing...")
    for txt in X_list:
        ids, types, attention = text_to_bert(tokenizer, txt)
        X_ids.append(ids)
        X_ids_types.append(types)
        X_attn_mask.append(attention)

    X_ids = np.array(X_ids, dtype=int)
    X_ids_types = np.array(X_ids_types, dtype=int)
    X_attn_mask = np.array(X_attn_mask, dtype=int)

    print("\nGenerated bert datasets:")
    print(f"X_ids shape is {X_ids.shape}")
    print(f"X_ids_types shape is {X_ids_types.shape}")
    print(f"X_attn_mask shape is {X_attn_mask.shape}")
    print(f"y shape is {y.shape}")
    return X_ids, X_ids_types, X_attn_mask, y