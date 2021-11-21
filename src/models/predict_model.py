from features.build_features import transform_dob_catage, get_columns_with_type, get_dummies_categorical
from utils import directory_path, load_pickle
import h2o
import pandas as pd

from data.make_dataset import generate_raw
from features.build_features import get_numerical_columns
from utils import get_metadata, get_model, load_pickle

def predict_h2o(model_type: str, version: str):
    """This function allows you to make predictions with H2O models

    :param model_type: model type
    :type model_type: str
    :param version: version of the model
    :type version: str
    """
    model = get_model(model_type, version)
    df = h2o.H2OFrame(preprocess_testset())
    prediction_id = pd.read_csv(f"{directory_path}data/raw/application_test.csv")['SK_ID_CURR']
    prediction = model.predict(df).as_data_frame().reset_index()
    pred = pd.concat([prediction_id,  prediction],  axis=1, ignore_index=True).rename(columns={0: "SK_ID_CURR", 1: "index", 2: "Predict", 3: "P0", 4:"P1" })
    pred = pred.drop(columns=['index'])
    pred.to_csv(f"{directory_path}/models/{version}/{model_type}/predictions.csv", index=False)

def predict_xgboost(version: str):
    """This function allows you to make predictions with xgboostClassifier Model

    :param model_type: model type
    :type model_type: str
    :param version: version of the model
    :type version: str
    """
    model = load_pickle(f"{directory_path}/models/{version}/xgboostClassifier/xgboostclassifier")
    df = preprocess_testset()
    prediction_id = pd.read_csv(f"{directory_path}data/raw/application_test.csv")['SK_ID_CURR']
    prediction = pd.Series(model.predict(df))
    pred = pd.concat([prediction_id,  prediction],  axis=1, ignore_index=True).rename(columns={0: "SK_ID_CURR", 1: "Prediction"})
    pred.to_csv(f"{directory_path}/models/{version}/xgboostClassifier/predictions.csv", index=False)

def preprocess_testset():
    """This function allows you to preprocess the test set

    :return: df
    :rtype: pd.DataFrame
    """
    generate_raw("application_test.csv")
    df = pd.read_csv(f"{directory_path}data/raw/application_test.csv")
    metadata = get_metadata()
    numerical_columns = get_numerical_columns(df, "TARGET")
    mode = metadata.get("mode")
    scaler = metadata.get("scaler")
    features_columns = metadata.get("features_columns")
    df = preprocess_numerical_features_test(df, numerical_columns, mode)
    df['cat_age'] = transform_dob_catage(df, 'DAYS_BIRTH')
    numerical_columns = [x for x in numerical_columns if x in features_columns]
    df = do_min_max_scaler_predict(df, numerical_columns, scaler)
    categorical_cols_test = get_columns_with_type(df, 'object')
    df = get_dummies_categorical(df, categorical_cols_test)
    df = df.reindex(columns=features_columns, fill_value=0)
    df = df[features_columns]
    df.to_csv(f"{directory_path}data/processed/application_test.csv", index=False)
    return df



def preprocess_numerical_features_test(df: pd.DataFrame, numerical_columns, mode):
    """This function allows you to fill NULL/NA values with the mode

    :param df: the dataframe to preprocessed
    :type df: pd.DataFrame
    :param numerical_columns: list of numerical columns in the dataframe
    :type numerical_columns: list
    :param mode: dictionary of numerical columns and associated mode
    :type mode: dict
    :return: the modified dataframe
    :rtype: pd.DataFrame
    """
    for feature in numerical_columns:
        if feature in mode:
            df[feature] = df[feature].fillna(mode[feature]).abs()
    return df


def do_min_max_scaler_predict(df: pd.DataFrame, numerical_columns, scaler):
    """This function allows you to transform features by scaling each feature to a given range

    :param df: dataframe to scale
    :type df:pd.DataFrame
    :param numerical_columns: list of numerical columns in the dataframe
    :type numerical_columns: list
    :param scaler: scaler
    :type scaler: MinMaxScaler
    :return: modified dataframe
    :rtype: pd.DataFrame
    """
    df[numerical_columns] = scaler.transform(df[numerical_columns])
    return df


def set_same_columns(df: pd.DataFrame, features_columns):
    columns = list(df.columns)
    for column in features_columns:
        if column not in columns:
            df[column] = 0
