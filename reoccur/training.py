import argparse
import openbayestool as obt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, matthews_corrcoef
import numpy as np
import joblib
import json

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True, choices=['main', 'alternate', 'realistic'],
                    help="Train and validate dataset")
parser.add_argument('-m', '--model', type=str, required=True,
                    choices=['RF', 'XGB', 'KNN', 'SVC', 'LR', 'GNB', 'DT'],
                    help="The machine learning algorithm")
parser.add_argument("--n_neighbors", type=float)
parser.add_argument("--weights", type=str)
parser.add_argument("--algorithm", type=str)
parser.add_argument("--leaf_size", type=float)
parser.add_argument("--p", type=float)
args = parser.parse_args()  # args即为获取的参数str形式

params = {
    "n_neighbors": int(args.n_neighbors),
    "weights": int(args.weights),
    "algorithm": args.algorithm,
    "leaf_size": args.leaf_size,
    "p": args.p
}


class Classifier:
    def __init__(self, classifier: str):
        if classifier == 'RF':
            self.model = RandomForestClassifier()
        elif classifier == 'XGB':
            self.model = XGBClassifier()
        elif classifier == 'KNN':
            self.model = KNeighborsClassifier(n_jobs=-1)
        elif classifier == 'SVC':
            self.model = SVC()
        elif classifier == 'LR':
            self.model = LogisticRegression()
        elif classifier == 'GNB':
            self.model = GaussianNB()
        elif classifier == 'DT':
            self.model = DecisionTreeClassifier()
        else:
            raise Exception("Classifier must be designated!")

        self.hyper_params = None

    def hyper_tuning(self, X_train, y_train, params: dict):
        search = GridSearchCV(estimator=self.model, param_grid=params,
                              scoring=['average_precision', 'accuracy', 'roc_auc', 'recall'], cv=5,
                              refit='average_precision',
                              n_jobs=-1, verbose=10)
        search.fit(X_train, y_train)
        self.hyper_params = search.best_params_
        return self.hyper_params

    def update_model(self, hyper_params: dict):
        """
        更新模型的超参数，生成一个未经训练的副本
        :param hyper_params: 最佳超参数
        :return:
        """
        self.model.set_params(**hyper_params)
        self.model = clone(self.model)

    def save_model(self, filename: str):
        joblib.dump(self.model, filename)

    def load_model(self, model_name: str):
        self.model = joblib.load(model_name)

    def predict(self, X, y):
        pred_y = self.model.predict(X)
        cm = confusion_matrix(y, pred_y)
        tn, tp, fn, fp = cm.ravel()
        return {
            "Sens": recall_score(y, pred_y),
            "Spec": tn / (tn + fp),
            "Acc": accuracy_score(y, pred_y),
            "AUC": roc_auc_score(y, pred_y),
            "MCC": matthews_corrcoef(y, pred_y),
        }

    def output_model_params(self, filename):
        json_data = json.dumps(self.hyper_params)
        with open(filename, "w") as f:
            f.write(json_data)

    def show_metrics(self, X, y):
        """
        使用5折交叉验证，分别输出训练时和验证时的平均指标
        :return:
        """
        # The scoring metrics
        scoring = {
            'Sens': make_scorer(lambda y, y_pred: confusion_matrix(y, y_pred)[1, 1] / (
                    confusion_matrix(y, y_pred)[1, 0] + confusion_matrix(y, y_pred)[1, 1])),
            'Spec': make_scorer(lambda y, y_pred: confusion_matrix(y, y_pred)[0, 0] / (
                    confusion_matrix(y, y_pred)[0, 0] + confusion_matrix(y, y_pred)[0, 1])),
            'Acc': 'accuracy',
            'AUC': 'roc_auc',
            'MCC': make_scorer(matthews_corrcoef)
        }

        # Perform 5-fold cross-validation
        cv_method = KFold(n_splits=5, shuffle=True, random_state=42)

        # Calculate performance metrics using cross_validate
        results = cross_validate(self.model, X, y, cv=cv_method, scoring=scoring)

        return {
            "test_Sens": np.mean(results["test_Sens"]),
            "test_Spec": np.mean(results["test_Spec"]),
            "test_Acc": np.mean(results["test_Acc"]),
            "test_AUC": np.mean(results["test_AUC"]),
            "test_MCC": np.mean(results["test_MCC"]),
        }


if __name__ == '__main__':
    # feature_generate("../dataset/Positive_realistic_dataset", "../dataset/pos_realistic_data.csv", 'aac')
    # feature_generate("../dataset/Negative_realistic_dataset", "../dataset/neg_realistic_data.csv", 'aac')

    # print("正在读取csv文件")
    # pos_df, neg_df = pd.read_csv('../dataset/pos_alternate_data.csv'), pd.read_csv("../dataset/neg_alternate_data.csv")
    # print("正在构造数据集标签")
    # pos_y, neg_y = np.array([1] * len(pos_df)), np.array([0] * len(neg_df))
    #
    # # Concat
    # print("正在拼接数据集")
    # X = np.array(pd.concat([pos_df, neg_df], axis=0))
    # y = np.concatenate([pos_y, neg_y])
    #
    # np.save("../dataset/alternate_data.npy", X)
    # np.save("../dataset/alternate_label.npy", y)

    print("Prepare to load data")
    X, y, search_params = None, None, None
    if args.dataset == 'alternate':
        X = np.load("/input0/alternate_data.npy")
        y = np.load("/input0/alternate_label.npy")
    elif args.dataset == 'realistic':
        X = np.load("/input0/realistic_data.npy")
        y = np.load("/input0/realistic_label.npy")

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    rfc = Classifier(args.model)
    rfc.model.set_params(**params)

    get_spec = lambda y, y_pred: confusion_matrix(y, y_pred)[0, 0] / (
            confusion_matrix(y, y_pred)[0, 0] + confusion_matrix(y, y_pred)[0, 1])

    for train_index, test_index in kf.split(X):
        # 划分训练集和测试集
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 训练模型
        rfc.model.fit(X_train, y_train)

        # 预测测试集
        y_pred = rfc.model.predict(X_test)

        sens = recall_score(y_test, y_pred)
        spec = get_spec(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        metrics = {
            "test_Sens": [],
            "test_Spec": [],
            "test_Acc": [],
            "test_AUC": [],
            "test_MCC": [],
        }

        metrics["test_Sens"].append(sens)
        metrics["test_Spec"].append(spec)
        metrics["test_Acc"].append(acc)
        metrics["test_AUC"].append(auc)
        metrics["test_MCC"].append(mcc)

    # best_params = rfc.hyper_tuning(X_train, y_train, search_params)

    obt.log_metric("Sens", np.mean(metrics['test_Sens']))
    obt.log_metric("Spec", np.mean(metrics['test_Spec']))
    obt.log_metric("Acc", np.mean(metrics['test_Acc']))
    obt.log_metric("AUC", np.mean(metrics['test_AUC']))
    obt.log_metric("MCC", np.mean(metrics['test_MCC']))

    print(metrics)

    rfc.save_model(args.algorithm + "_" + args.dataset + ".model")
