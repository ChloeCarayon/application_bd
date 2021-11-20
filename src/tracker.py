import pandas as pd
import h2o
import sys
from sklearn.model_selection import train_test_split
from h2o.estimators.xgboost import H2OXGBoostEstimator

import mlflow
from models.train_model import define_h2o_dataframe, define_features, do_sampling
from data.make_dataset import generate_raw
from features.build_features import generate_features
from utils import directory_path


def get_preprocessed_h2o_dataset():
    """This function allows you to generate
    the dataset for training and validation.
    - get the raw data
    - do preprocessing, feature engineering, dowsampling
    - split in train and valid set

    :return df_h2o_train: train dataset
    :rtype df_h2o_train: h2o.H2OFrame
    :return df_h2o_valid: valid dataset
    :rtype df_h2o_valid: h2o.H2OFrame
    :return features: list of features columns
    :rtype features:
    :return label: target
    :rtype label: str
    """
    try:
        df_path = f"{directory_path}/data/processed/application_train.csv"
        df = pd.read_csv(df_path)
    except Exception:
        generate_raw("application_train.csv")
        generate_features(directory_path, 'TARGET')
        df_path = f"{directory_path}/data/processed/application_train.csv"
        df = pd.read_csv(df_path)

    label = 'TARGET'
    df = do_sampling(df, label)
    df_h2o_train, df_h2o_valid = define_train_test_validation(df, label)
    features = define_features(df_h2o_train, label)
    return df_h2o_train, df_h2o_valid, features, label


def define_train_test_validation(df: pd.DataFrame, label: str):
    """This function split the dataset in stratified
    train and valid set and transform them in h2o frame

    :param df: dataset preprocessed
    :type df: pd.DataFrame
    :param label: target
    :type label: str
    :return df_h2o_train: train dataset
    :rtype df_h2o_train: h2o.H2OFrame
    :return df_h2o_valid: valid dataset
    :rtype df_h2o_valid: h2o.H2OFrame
    """
    df_train, df_valid = train_test_split(df, stratify = df[label], test_size=0.2)
    df_h2o_train = define_h2o_dataframe(df_train, label)
    df_h2o_valid = define_h2o_dataframe(df_valid, label)
    return df_h2o_train, df_h2o_valid


def train_xgboost_model(df_h2o_train: h2o.H2OFrame, df_h2o_validation: h2o.H2OFrame,
                        features, label: str, ntrees:int = 200, max_depth:int = 10,
                        learn_rate:float = 0.01, min_rows:int= 5 ):
    """This function train xgboost model.

    :param df_h2o_train: train dataset
    :type df_h2o_train: h2o.H2OFrame
    :param df_h2o_valid: valid dataset
    :type df_h2o_valid: h2o.H2OFrame
    :param features: list of features columns
    :type features:
    :param label: target
    :type label: str
    :param ntrees: number of trees for xgboost model
    :type ntrees: int
    :param max_depth: max depth for xgboost model
    :type max_depth: int
    :param learn_rate: learning rate for xgboost model
    :type learn_rate: float
    :param min_rows: min rows for xgboost model
    :type min_rows: int
    """

    with mlflow.start_run(run_name="Experimentation"):
        xgboost = H2OXGBoostEstimator(ntrees= ntrees, max_depth=max_depth,
                                      learn_rate=learn_rate, min_rows=min_rows,
                                      sample_rate= 0.9, col_sample_rate_per_tree=0.9,
                                      score_tree_interval= 10
                                      )


        xgboost.train(features, label, training_frame=df_h2o_train, validation_frame=df_h2o_validation)

        mlflow.log_params({"ntrees": ntrees,
                           "max_depth": max_depth,
                           "learn_rate": learn_rate,
                           "min_rows": min_rows
                           })

        mlflow.log_metrics(eval_metrics(xgboost))

        mlflow.h2o.log_model(xgboost, "model")

        mlflow.tracking.get_tracking_uri()
        mlflow.end_run()

def eval_metrics(xgboost):
    """This function evaluate the model metrics.

    :param xgboost: xgboost model
    :type xgboost:
    :return metrics: dictionary containing the metrics
    :rtype metrics:
    """
    f1 = xgboost.F1()[0][1]
    f05 = xgboost.F0point5()[0][1]
    f2 = xgboost.F2()[0][1]
    auc = xgboost.auc()
    mcc = xgboost.mcc()[0][1]
    aucpr = xgboost.aucpr()

    metrics = {
        "f1-score": f1,
        "f0.5-score": f05,
        "f2-score": f2,
        "auc": auc,
        "mcc": mcc,
        "aucpr": aucpr,
    }
    return metrics

def train_mlflow(ntrees: int = 200, max_depth:int = 10,
                 learn_rate:float = 0.01, min_rows:int = 5):
    """This function trains the model given hyperparameters.

    :param ntrees: number of trees for xgboost model
    :type ntrees: int
    :param max_depth: max depth for xgboost model
    :type max_depth: int
    :param learn_rate: learning rate for xgboost model
    :type learn_rate: float
    :param min_rows: min rows for xgboost model
    :type min_rows: int
    """
    h2o.init()
    df_h2o_train, df_h2o_valid, features, label = get_preprocessed_h2o_dataset()
    train_xgboost_model(df_h2o_train, df_h2o_valid, features,label,
                        ntrees, max_depth,
                        learn_rate, min_rows)
    h2o.cluster().shutdown()

if __name__ == "__main__":
    ntrees = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    learn_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    min_rows = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    train_mlflow(ntrees, max_depth,
                 learn_rate, min_rows)