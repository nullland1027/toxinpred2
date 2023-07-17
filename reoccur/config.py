random_forest_params = {
    'n_estimators':               [50, 100, 150, 200],
    'criterion':                  ["gini", "entropy", "log_loss"],
    'max_depth':                  [10, 20, None],
    'min_samples_split':          [1, 2, 5],
    # 'min_samples_leaf':         [1],
    # 'min_weight_fraction_leaf': [0.0],
    'max_leaf_nodes':             [10, 50, None],

}

xgboost_params = {
    'n_estimators':     [100, 200, 500],
    'max_depth':        [20, 50, 100, None],
    'learning_rate':    [0.01, 0.05, 0.1],
    'subsample':        [0.1, 0.5, 0.8],
    'min_child_weight': [0.1, 1, 10, 100],
    'colsample_bytree': [0.1, 0.5, 0.8],
    'reg_alpha':        [0.1, 1, 10],
    'reg_lambda':       [0.1, 1, 10],
}

knn_params = {
    'n_neighbors':      [2, 5, 10, 20],
    'weights':          ['uniform', 'distance'],
    'algorithm':        ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size':        [10, 30, 50, 100],
    'p':                [1, 2]
}

support_vector_classifier_params = {
    'C':      [1, 10, 50],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    'degree': [3, 4],
    'gamma':  [],
    'coef0':  []
}


logistic_regression_params = {
    'penalty':          ['l1', 'l2', 'elasticnet'],
    'tol':              [1e-4, 1e-5],
    'C':                [0.1, 1, 2, 5, 10],
    'solver':           ['saga'],
    'l1_ratio':         [0, 0.1, 0.4, 0.6, 0.9, 1],
    'max_iter':         [100, 200, 500],
}

gaussian_naive_bayes_params = {

}

decision_tree_params = {
    'criterion':                  ["gini", "entropy", "log_loss"],
    'max_depth':                  [10, 50, 100, 500, None],
    'min_samples_split':          [1, 2, 5, 10, 20, 50],
    'min_samples_leaf':           [1, 2, 5, 10, 20, 50],
    'min_weight_fraction_leaf':   [0.0, 0.1, 0.5, 0.8, 1.0]
}
