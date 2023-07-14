from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix, matthews_corrcoef
import numpy as np


class Classifier:
    def __init__(self, classifer: str):
        if classifer == 'RF':
            self.model = RandomForestClassifier()
        elif classifer == 'XGB':
            self.model = XGBClassifier()
        elif classifer == 'KNN':
            self.model = KNeighborsClassifier()
        elif classifer == 'SVC':
            self.model = SVC()
        elif classifer == 'LR':
            self.model = LogisticRegression()
        elif classifer == 'GNB':
            self.model = GaussianNB()
        elif classifer == 'DT':
            self.model = DecisionTreeClassifier()
        else:
            raise Exception("Classifer must be designated!")

    def hyper_tuning(self, X_train, y_train, params: dict):
        search = GridSearchCV(estimator=self.model, param_grid=params,
                              scoring=['average_precision', 'accuracy', 'roc_auc', 'recall'], cv=5, refit='average_precision',
                              n_jobs=-1, verbose=10)
        search.fit(X_train, y_train)
        return search.best_params_

    def update_model(self, hyper_params: dict):
        self.model.set_params(**hyper_params)

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
        cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Calculate performance metrics using cross_val_score
        results = cross_val_score(self.model, X, y, cv=cv_method, scoring=scoring)

        # Calculate the average of each metric across the folds
        avg_results = {metric: np.mean(results[metric]) for metric in results.keys()}
        return avg_results
