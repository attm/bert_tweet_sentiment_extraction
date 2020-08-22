import os
import numpy as np
import pandas as pd


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