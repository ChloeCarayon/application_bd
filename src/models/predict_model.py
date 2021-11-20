from features.build_features import transform_dob_catage, get_columns_with_type, get_dummies_categorical
from utils import directory_path, load_pickle
import h2o
import pandas as pd

from data.make_dataset import generate_raw
from features.build_features import get_numerical_columns


def predict(model_type: str, version: str):
    """This function allows you to make predictions

    :param model_type: model type
    :type model_type: str
    :param version: version of the model
    :type version: str
    """
    model = get_model(model_type, version)
    df = preprocess_testset()
    model.predict(df)


def preprocess_testset():
    """This function allows you to preprocess the test set

    :return: df_h2o
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
    df_h2o = h2o.H2OFrame(df)
    return df_h2o


def get_model(model_type: str, version: str):
    """This function allows you to get the model

    :param model_type: model type
    :type model_type: str
    :param version: version of the model
    :type version: str
    :return: model
    :rtype:
    """
    path_model = f"{directory_path}models/{version}/{model_type}.zip"
    model = h2o.import_mojo(path_model)
    return model


def get_metadata():
    """This function allows you to get the metadata

    :return: metadata
    :rtype:
    """
    path_metadata = f"{directory_path}models/metadata"
    metadata = load_pickle(path_metadata)
    return metadata


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
