MODELS_PARAMS = {
    "Xgboost": {
        "ntrees": 150,
        "max_depth": 5,
        "learn_rate": 0.1,
        "sample_rate": 0.9,
        "col_sample_rate_per_tree": 0.9,
        "min_rows": 5,
        "stopping_rounds": 5,
        "stopping_metric": "AUC",
        "seed": 1,
        "score_tree_interval": 10,
    },
    "GradientBoosting": {
        "nfolds": 5,
        "ntrees": 100,
        "seed": 1,
        "max_depth": 9,
        "stopping_rounds": 5,
        "stopping_metric": "auc",
    },
    "RandomForest": {
        "nfolds": 5,
        "ntrees": 100,
        "seed": 1,
        "max_depth": 9,
        "stopping_rounds": 5,
        "stopping_metric": "AUC",
    },
    "XgboostClassifier": {
        "n_estimators": 150,
        "max_depth": 5,
        "min_child_weight": 1,
        "learning_rate": 0.1,
        "objective":'binary:logistic',
        "colsample_bytree": 0.8,
        "min_child_weight": 1
    },
}
