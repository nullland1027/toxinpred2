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


class Classifier:
    def __init__(self, classifier: str):
        if classifier == 'RF':
            self.model = RandomForestClassifier()
        elif classifier == 'XGB':
            self.model = XGBClassifier()
        elif classifier == 'KNN':
            self.model = KNeighborsClassifier()
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
                              n_jobs=-1, verbose=4)
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
