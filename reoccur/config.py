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
    'subsample':        [0.1, 0.5],  # for imbalanced data
    'min_child_weight': [0.1, 1, 10, 100],
    'max_delta_step':   [0, 0.1, 0.5, 1],  # for imbalanced data
    'scale_pos_weight': [10],  # for imbalanced data
    'colsample_bytree': [0.1, 0.5],  # for imbalanced data
    'reg_alpha':        [10],  # for imbalanced data
    'reg_lambda':       [10],  # for imbalanced data
}

knn_params = {
    'n_neighbors':      [2, 5, 10, 20],
    'weights':          ['uniform', 'distance'],
    'algorithm':        ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size':        [10, 30, 50, 100],
    'p':                [1, 2]
}

support_vector_classifier_params = {
    'C':      [1, 10, 50, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    'degree': [3, 4, 5, 8, 10, 20],
    'gamma':  ['scale', 'auto'],
    'coef0':  [-0.8, -0.5, 0.0, 0.5, 0.8],

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
