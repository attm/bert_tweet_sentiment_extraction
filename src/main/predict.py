import os 
from os.path import join as pjoin
import numpy as np
import pandas as pd
from src.model.bert_semantic_classifier import build_bert_model


cwd = os.getcwd()
BERT_DATASETS_FOLDER_PATH = pjoin(cwd, "data/processed/bert_datasets")
BERT_SAVED_CP = pjoin(cwd, "saved_models/bert")
SAMPLE_SUBMISSION_CSV_PATH = pjoin(cwd, "data/raw/sample_submission.csv")
RESUL_SUBMISSION_CSV_PATH = pjoin(cwd, "data/raw/result_submission.csv")


def load_data(datasets_folder : str) -> np.ndarray:
    X_test_ids = np.load(pjoin(datasets_folder, "X_test_ids.npy"))
    X_test_ids_types = np.load(pjoin(datasets_folder, "X_test_ids_types.npy"))
    X_test_attn_mask = np.load(pjoin(datasets_folder, "X_test_attn_mask.npy"))
    return X_test_ids, X_test_ids_types, X_test_attn_mask

def main():
    X_test_ids, X_test_ids_types, X_test_attn_mask = load_data(BERT_DATASETS_FOLDER_PATH)
    print(f"\nLoaded datasets from {BERT_DATASETS_FOLDER_PATH}")
    print(f"Shape of X_test_ids is {X_test_ids.shape}")

    bert_model = build_bert_model()
    bert_model.load_weights(BERT_SAVED_CP)
    y = bert_model.predict([X_test_ids, X_test_ids_types, X_test_attn_mask])
    y_pred = np.argmax(y, axis=1)
    y_pred = y_pred.tolist()

    df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
    df['sentiment'] = y_pred
    df.to_csv(RESUL_SUBMISSION_CSV_PATH, index=False)

if __name__ == "__main__":
    main()
else:
    raise ImportError("Predict is the main script, shouldn't be imported.")