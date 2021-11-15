import click
import h2o
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from sklearn.utils import resample
from h2o.automl import H2OAutoML

from src.models.model_parameters import MODELS_PARAMS

h2o.init()


def do_sampling(df: pd.DataFrame, label: str, downsample_coef: float = 0.75):
    """This function allows you to balance unbalanced dataset.
    By specifying the downsample coefficient value,
    one can downsample its dataset.

    :param df: dataset to downsample
    :type df: pd.DataFrame
    :param label: target
    :type label: str
    :param downsample_coef: dowsampling coefficient to apply
    :type downsample_coef: float
    """
    df_majority = df[df[label] == 0]
    df_minority = df[df[label] == 1]
    samples_to_keep = df_majority - df_majority * downsample_coef
    df_majority_downsampled = resample(
        df_majority, replace=False, n_samples=samples_to_keep
    )
    df_up_down_sampled = pd.concat([df_majority_downsampled, df_minority])
    return df_up_down_sampled


def define_h2o_dataframe(df: pd.DataFrame, label: str):
    """This function transforms a pandas DataFrame
    into H2O DataFrame.
    It also defines the target column.

    :param df: dataset
    :type df: pd.DataFrame
    :param label: target
    :type label: str
    """
    df_h2o = h2o.H2OFrame(df)
    df_h2o[label].asfactor().levels()
    df_h2o[label] = df_h2o[label].asfactor()
    return df_h2o


def define_features(df: h2o.frame.H2OFrame, label: str):
    """This function returns the feature columns
    in a list without the label column.

    :param df: dataset
    :type df: h2o.frame.H2OFrame
    :param label: target
    :type label: str
    """
    features = df.col_names
    features.remove(label)
    return features




def split_dataset(
    df: h2o.frame.H2OFrame, ratio_train_test: float, ratio_validation: float
):
    """This function splits dataset
    in train, test, validation.

    :param df: dataset to downsample
    :type df: h2o.frame.H2OFrame
    :param ratio_train_test: ratio for train/test
    :type ratio_train_test: float
    :param ratio_validation: ratio for validation
    :type ratio_validation: float
    """
    train, valid, test = df.split_frame(ratios=[ratio_train_test, ratio_validation])
    return train, valid, test


def select_model(model_type: str):
    """This function selects and return the
    corresponding model.

    :param model_type: model type
    :type model_type: str
    """
    if model_type == "xgb":
        return H2OXGBoostEstimator(**MODELS_PARAMS["Xgboost"])
    elif model_type == "gbm":
        return H2OGradientBoostingEstimator(**MODELS_PARAMS["GradientBoosting"])
    elif model_type == "automl":
        return H2OAutoML(max_models =25, balance_classes=True, seed =1)
    else:
        return H2ORandomForestEstimator(**MODELS_PARAMS["RandomForest"])


def do_training(model_type: str, label: str, dataset_path: str):
    """This function:
    - gets the model type
    - gets the dataset and transform it in h2o format
    - trains model
    - return model

    :param model_type: model type
    :type model_type: str
    :param label: target column's name
    :type label: str
    :param dataset_path: path to dataset
    :type dataset_path: str
    """
    model = select_model(model_type)

    df = pd.read_csv(dataset_path)

    df_h2o = define_h2o_dataframe(df, label)
    features = define_features(df, label)

    train, valid, test = split_dataset(df_h2o, 0.8, 0.1)
    model.train(x=features, y=label, training_frame=train, validation_frame=valid)

    return model


@click.command()
@click.option(
    "--model_type",
    prompt="Choose a model to train",
    type=click.Choice(["xgboost", "gradient boosting", "random forest"]),
)
def train_model(model_type):
    label = "TARGET"
    dataset_path = "../data/processed/train.csv"
    model = do_training(model_type, label, dataset_path)
    if model_type == "automl":
        model = model.get_best_model()
    model_path = f"../models/{model_type}.zip"
    model.download_mojo(model_path)
