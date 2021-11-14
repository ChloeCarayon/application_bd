MODELS_PARAMS = {
    "Xgboost": {
        "ntrees": 200,
        "max_depth": 10,
        "learn_rate": 0.01,
        "sample_rate": 0.9,
        "col_sample_rate_per_tree": 0.9,
        "min_rows": 5,
        "seed": 1,
        "score_tree_interval": 10,
    },
    "GradientBoosting": {
        "nfolds": 5,
        "ntrees": 100,
        "seed": 1,
        "max_depth": 9,
        "stopping_rounds": 5,
        "stopping_metric": "AUC",
        "balance_classes": True,
    },
    "RandomForest": {
        "nfolds": 5,
        "ntrees": 100,
        "seed": 1,
        "max_depth": 9,
        "stopping_rounds": 5,
        "stopping_metric": "AUC",
        "balance_classes": True,
    },
}
