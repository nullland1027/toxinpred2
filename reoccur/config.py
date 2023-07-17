random_forest_params = {
    'n_estimators':               [50, 100, 150, 200],
    'criterion':                  ["gini", "entropy", "log_loss"],
    'max_depth':                  [10, 20, None],
    'min_samples_split':          [1, 2, 5],
    # 'min_samples_leaf':         [1],
    # 'min_weight_fraction_leaf': [0.0],
    'max_leaf_nodes':             [10, 50, None],

}

logistic_regression_params = {
    'penalty':  ['l1', 'l2', 'elasticnet'],
    'tol':      [1e-4, 1e-5],
    'C':        [0.1, 1, 2, 5, 10],
    'solver':   ['saga'],
    'l1_ratio': [0, 0.1, 0.4, 0.6, 0.9, 1],
    'max_iter': [100, 200, 500]
}

