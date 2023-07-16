import numpy as np
import pandas as pd
from Pfeature import pfeature
from classifier import Classifier
from sklearn.utils import shuffle
from config import random_forest_params


def feature_generate(input_file: str, output_file: str, method: str):
    if method == 'aac':
        pfeature.aac_wp(file=input_file, out=output_file)
    elif method == 'PSSM':
        pass


if __name__ == '__main__':
    # feature_generate("../dataset/Positive_realistic_dataset", "../dataset/pos_realistic_data.csv", 'aac')
    # feature_generate("../dataset/Negative_realistic_dataset", "../dataset/neg_realistic_data.csv", 'aac')

    # print("正在读取csv文件")
    # pos_df, neg_df = pd.read_csv('../dataset/pos_realistic_data.csv'), pd.read_csv("../dataset/neg_realistic_data.csv")
    # print("正在构造数据集标签")
    # pos_y, neg_y = np.array([1] * len(pos_df)), np.array([0] * len(neg_df))
    #
    # # Concat
    # print("正在拼接数据集")
    # X = np.array(pd.concat([pos_df, neg_df], axis=0))
    # y = np.concatenate([pos_y, neg_y])
    #
    # np.save("../dataset/realistic_data.npy", X)
    # np.save("../dataset/realistic_label.npy", y)

    print("加载数据")
    X = np.load("../dataset/realistic_data.npy")
    y = np.load("../dataset/realistic_label.npy")

    print("准备数据集随机打乱")
    X_train, y_train = shuffle(X, y, random_state=42)

    rfc = Classifier('RF')

    best_params = rfc.hyper_tuning(X_train, y_train, random_forest_params)
    rfc.output_model_params("RF_realistic_params.json")
    print("Best params", best_params)

    print("更新模型超参数")
    rfc.update_model(best_params)

    print(rfc.show_metrics(X_train, y_train))

    rfc.save_model("RF_realistic.model")
