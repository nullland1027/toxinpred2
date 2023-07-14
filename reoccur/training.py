import numpy as np
import pandas as pd
from Pfeature import pfeature
from classifier import Classifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from config import random_forest_params


def feature_generate(input_file: str, output_file: str, method: str):
    if method == 'acc':
        pfeature.aac_wp(file=input_file, out=output_file)
    elif method == 'PSSM':
        pass


if __name__ == '__main__':
    # feature_generate("../dataset/Positive_alternate_dataset", "../dataset/pos_alternate_data.csv", 'acc')
    # feature_generate("../dataset/Negative_alternate_dataset", "../dataset/neg_alternate_data.csv", 'acc')

    pos_df, neg_df = pd.read_csv('../dataset/pos_alternate_data.csv'), pd.read_csv("../dataset/neg_alternate_data.csv")
    pos_y, neg_y = np.array([1] * len(pos_df)), np.array([0] * len(neg_df))

    # Concat
    X = np.array(pd.concat([pos_df, neg_df], axis=0))
    y = np.concatenate([pos_y, neg_y])

    X_train, y_train = shuffle(X, y, random_state=42)

    rfc = Classifier('RF')

    best_params = rfc.hyper_tuning(X_train, y_train, random_forest_params)
    rfc.update_model(best_params)
    print(rfc.show_metrics(X_train, y_train))
