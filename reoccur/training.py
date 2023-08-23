import numpy as np
import argparse
import os
from Pfeature import pfeature
import pandas as pd
# from Pfeature import pfeature
from classifier import Classifier
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, matthews_corrcoef
from config import random_forest_params, logistic_regression_params, decision_tree_params
from config import xgboost_params, knn_params, support_vector_classifier_params, gaussian_naive_bayes_params

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True, choices=['main', 'alternate', 'realistic'],
                    help="Train and validate dataset")
parser.add_argument('-a', '--algorithm', type=str, required=True,
                    choices=['RF', 'XGB', 'KNN', 'SVC', 'LR', 'GNB', 'DT'],
                    help="The machine learning algotithm")
args = parser.parse_args()  # args即为获取的参数str形式

# def feature_generate(input_file: str, output_file: str, method: str):
#     if method == 'aac':
#         pfeature.aac_wp(file=input_file, out=output_file)
#     elif method == 'PSSM':
#         pass


if __name__ == '__main__':
    # feature_generate("../dataset/Positive_realistic_dataset", "../dataset/pos_realistic_data.csv", 'aac')
    # feature_generate("../dataset/Negative_realistic_dataset", "../dataset/neg_realistic_data.csv", 'aac')

    print("正在读取csv文件")
    pos_df, neg_df = pd.read_csv('../dataset/pos_alternate_data.csv'), pd.read_csv("../dataset/neg_alternate_data.csv")
    print("正在构造数据集标签")
    pos_y, neg_y = np.array([1] * len(pos_df)), np.array([0] * len(neg_df))

    # Concat
    print("正在拼接数据集")
    X = np.array(pd.concat([pos_df, neg_df], axis=0))
    y = np.concatenate([pos_y, neg_y])

    np.save("../dataset/alternate_data.npy", X)
    np.save("../dataset/alternate_label.npy", y)

    print("Prepare to load data")
    X, y, search_params = None, None, None
    if args.dataset == 'alternate':
        X = np.load("../dataset/feature_acc/alternate_data.npy")
        y = np.load("../dataset/feature_acc/alternate_label.npy")
    elif args.dataset == 'realistic':
        X = np.load("../dataset/feature_acc/realistic_data.npy")
        y = np.load("../dataset/feature_acc/realistic_label.npy")

    if args.algorithm == 'RF':
        search_params = random_forest_params
    elif args.algorithm == 'XGB':
        search_params = xgboost_params
    elif args.algorithm == 'KNN':
        search_params = knn_params
    elif args.algorithm == 'SVC':
        search_params = support_vector_classifier_params
    elif args.algorithm == 'LR':
        search_params = logistic_regression_params
    elif args.algorithm == 'GNB':
        search_params = gaussian_naive_bayes_params
    elif args.algorithm == 'DT':
        search_params = decision_tree_params

    print("Prepare to shuffle dataset")
    X_train, y_train = shuffle(X, y, random_state=42)

    rfc = Classifier(args.algorithm)

    best_params = rfc.hyper_tuning(X_train, y_train, search_params)

    rfc.output_model_params(os.path.join("params", args.algorithm + "_" + args.dataset + "_" + "_params.json"))
    print("Best params", best_params)

    print("Update model hyper params")
    rfc.update_model(best_params)

    print(rfc.show_metrics(X_train, y_train))

    rfc.save_model(os.path.join("models", args.algorithm + "_" + args.dataset + ".model"))
