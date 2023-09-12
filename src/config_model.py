MODEL_PARAMS = {
    'MLP Classifier': {
        'hidden_layer_sizes':[(20, 10, 5), (30, 15, 10, 5), 
                            (20, 5, 10), (20, 10, 5, 10)],
        'activation': ['logistic', 'tanh'],
        'solver': ['lbfgs', 'sgd', 'adam'], 
        'alpha': [1e-1, 1e-2, 1e-3, 1e-4],
        'batch_size': [5, 10, 50],
        'learning_rate': ['constant', 'invscaling'],
        'learning_rate_init': [0.01, 0.001, 0.0001]
    },
    'Logistic Regression': {
        'penalty': ['l2', None], 
        'tol': [1e-1, 1e-2, 1e-3, 1e-4],
        'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'],
        'max_iter': [100, 200], # [100, 200, 300],
        'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    },
    'KNN': {
        'n_neighbors': [3, 4, 5, 6, 7, 8],  
        'weights': ['uniform'], 
        'leaf_size': [10, 20, 30, 40], 
        'metric': ['minkowski']
    },
    'Decision Tree': {
        'criterion': ['gini', 'entropy', 'log_loss'], 
        'max_depth': [5, 10, 20], 
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [2, 3, 4, 5], 
        'splitter': ['best', 'random'],
        'class_weight': [None, 'balanced'],
    },
    'SVC': {
        'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4, 5, 6],
        'gamma': ['scale', 'auto'],
        'tol': [1e-1, 1e-2, 1e-3, 1e-4],
        'class_weight': [None, 'balanced']
    },
    'Random Forest': {
        'n_estimators':[50, 100, 150], 
        'criterion': ['gini', 'entropy', 'log_loss'], 
        'min_samples_split': [2, 3, 4, 5], 
        'min_samples_leaf': [1, 2, 3]
    },
    'XGBoost': {
        'max_depth': [3, 4, 5, 7],
        'base_score': [0.5], 
        'booster': ['gbtree'],
        'learning_rate': [0.1, 0.01, 0.05],
        'n_estimators': [50, 100],
        'reg_alpha': [0], 
        'reg_lambda': [0, 1, 10],
        'gamma': [0, 0.25, 1],
        'subsample': [0.8],
        'colsample_bytree': [0.5]
    },
    'LightGBM': {
        'num_leaves': [2, 3, 5],
        'max_depth': [3, 5, 7, 9],
        'n_estimators': [50, 100],
        'learning_rate':[0.01, 0.03, 0.1, 0.3],
        'reg_lambda': [10, 30, 60],
        'reg_alpha': [10, 30, 60],
        'min_gain_to_split': [2, 5, 10]
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 150, 200], 
        'learning_rate': [0.01, 0.03, 0.1, 0.3, 1.0, 10]
    }
}

MULTILABEL_MODEL_PARAMS = {
    'Logistic Regression': {
        'penalty': ['l2'], 
        'tol': [1e-1, 1e-2, 1e-3, 1e-4],
        'solver': ['lbfgs', 'sag'],
        'max_iter': [100],
        'C': [0.05, 0.1, 0.5, 1, 5],
    },
    'XGBoost': {
        'max_depth': [4, 5],
        'base_score': [0.5], 
        'booster': ['gbtree'],
        'learning_rate': [0.1, 0.01, 0.05],
        'n_estimators': [50, 100],
        'reg_alpha': [0, 0.5, 1], 
        'reg_lambda': [0, 1, 10],
        'gamma': [0, 0.25, 1],
        'subsample': [0.8],
        'colsample_bytree': [0.5]
    },
    'SVC': {
        'C': [0.01, 0.05, 0.1, 0.5, 1, 5],
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'degree': [3, 4, 5, 6],
        'gamma': ['auto'],
        'tol': [1e-1, 1e-2, 1e-3, 1e-4],
        'class_weight': ['balanced']
    },
    'Random Forest': {
        'n_estimators':[50, 100, 150], 
        'criterion': ['gini', 'entropy', 'log_loss'], 
        'min_samples_split': [2, 3, 4, 5], 
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt']
    }
}

