import h2o
import pandas as pd
from h2o.automl import H2OAutoML
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from sklearn.utils import resample

from models.model_parameters import MODELS_PARAMS
from utils import directory_path, define_h2o_dataframe, dump_pickle, dump_json_object
#from visualization.do_explicability import do_visualisation
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, classification_report


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
    :return: a new df.DataFrame which has been downsample
    :rtype: df.DataFrame
    """

    df_majority = df[df[label] == 0]
    df_minority = df[df[label] == 1]
    #samples_to_keep = int(df_majority.shape[0] - df_majority.shape[0] * downsample_coef)
    df_majority_downsampled = resample(
        df_majority, replace=True,
        n_samples=40000, random_state=1
    )
    df_up_down_sampled = pd.concat([df_majority_downsampled, df_minority])
    return df_up_down_sampled


def define_features(df: h2o.H2OFrame, label: str):
    """This function returns the feature columns
    in a list without the label column.

    :param df: dataset
    :type df: h2o.H2OFrame
    :param label: target
    :type label: str
    :return: a list of the features for the model
    :rtype: list(str)
    """
    features = df.col_names
    features.remove(label)
    return features


def split_dataset_h2o(df: pd.DataFrame, label: str, ratio_validation: float = 0.2):
    """This function splits dataset
    in train, validation h2o dataframe.

    :param df: dataset to downsample
    :type df: pd.DataFrame
    :param label: target
    :type label: str
    :param ratio_train_test: ratio for train/validation
    :type ratio_train_test: float
    :return: three new dataset (train, valid and test)
    :rtype: h2o.H2OFrame
    """
    df_train, df_valid = train_test_split(df, stratify=df[label], test_size=ratio_validation, random_state=42)

    df_train.to_csv(f"{directory_path}visualisation/h2o_xgboost/train_set.csv", index=False)

    df_train_h2o =define_h2o_dataframe(df_train, label)
    df_valid_h2o =define_h2o_dataframe(df_valid, label)
    return df_train_h2o, df_valid_h2o


def select_model(model_type: str):
    """This function selects and return the
    corresponding model.

    :param model_type: model type
    :type model_type: str
    :return: a specific model of H2O
    :rtype: models
    """
    print(model_type)
    if model_type == "xgboost":
        return H2OXGBoostEstimator(**MODELS_PARAMS["Xgboost"])
    elif model_type == "gmb":
        return H2OGradientBoostingEstimator(**MODELS_PARAMS["GradientBoosting"])
    elif model_type == "automl":
        return H2OAutoML(max_models=25, balance_classes=True, seed=1)
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
    :return: model and test set
    :rtype: models and h2o.H2OFrame dataset
    """
    model = select_model(model_type)
    print(type(model))
    df = pd.read_csv(dataset_path)
    df = do_sampling(df, label)
    train, valid = split_dataset_h2o(df, label, 0.2)
    features = define_features(train, label)
    model.train(x=features, y=label, training_frame=train, validation_frame=valid)

    perf = model.model_performance(valid)
    metrics = {
        "f1" : model.F1()[0][1],
        "f05" : model.F0point5()[0][1],
        "f2" : model.F2()[0][1],
        "auc" : model.auc(),
        "mcc" : model.mcc()[0][1],
        "aucpr" : model.aucpr()
    }

    return model, metrics



def train_model_h2o(model_type, version):
    """This function:
    - get the preprocessed dataset
    - train the correct model
    - store it in mojo format

    :param model_type: model type
    :type model_type: str
    :param version: version to store model
    :type version: str
    """
    label = "TARGET"
    dataset_path = f"{directory_path}/data/processed/application_train.csv"
    model, metrics = do_training(model_type, label, dataset_path)
    if model_type == "automl":
        model = model.get_best_model()
    model_path = f"{directory_path}/models/{version}/{model_type}.zip"
    model.download_mojo(model_path)
    dump_json_object(f"{directory_path}/models/{version}/{model_type}_metrics.json", metrics)


def train_model_xboost(version):
    label = 'TARGET'
    dataset_path = f"{directory_path}/data/processed/application_train.csv"
    df = pd.read_csv(dataset_path)
    df = do_sampling(df, label)
    df_train, df_test = train_test_split(df, stratify=df[label], test_size=0.2)

    model = XGBClassifier(**MODELS_PARAMS['XgboostClassifier'])
    model.fit(df_train.drop(columns=[label]), df_train[label])
    prediction = model.predict(df_test.drop(columns=[label]))
    metrics = {
        "f1score": f1_score(df_test[label], prediction, average='macro'),
        "auc":roc_auc_score(df_test[label], prediction),
        "mcc": matthews_corrcoef(df_test[label], prediction),
        "classification_report": classification_report(df_test[label], prediction)
    }
    print(metrics)
    dump_json_object(f"{directory_path}/models/{version}/xgboostclassifier_metrics.json", metrics)
    dump_pickle(f"{directory_path}/models/{version}/xgboostclassifier", model)
    df_train.to_csv(f"{directory_path}visualisation/xgboost/train_set.csv", index=False)
