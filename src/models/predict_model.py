from src.features.build_features import transform_dob_catage, get_columns_with_type, get_dummies_categorical
from src.utils import directory_path, load_pickle
import h2o
import pandas as pd

from src.data.make_dataset import generate_raw
from src.features.build_features import get_numerical_columns


def predict(model, version):
    df = generate_raw("application_test.csv")
    model, metadata = get_model_metadata(model, version)
    numerical_columns = get_numerical_columns(df)
    mode = metadata.get("mode")
    scaler = metadata.get("scaler")
    features_columns = metadata.get("features_columns")
    df = preprocess_numerical_features_test(df, numerical_columns, mode)
    df['cat_age'] = transform_dob_catage(df, 'DAYS_BIRTH')
    df = df.drop(columns=['DAYS_BIRTH'])
    do_min_max_scaler_predict(df, numerical_columns, scaler)
    categorical_cols_test = get_columns_with_type(df, 'object')
    df = get_dummies_categorical(df, categorical_cols_test)
    df = df.reindex(columns=features_columns, fill_value=0)
    return df


def get_model_metadata(model_type, version):
    """This function allows you to get the model and the metadata

    :param model_type: model type
    :type model_type: str
    :param version: version of the model
    :type version: int
    :return: model
    :rtype:
    :return: metadata
    :rtype:
    """
    path_model = f"{directory_path}/models/{version}/{model_type}.zip"
    model = h2o.import_mojo(path_model)
    path_metadata = f"{directory_path}/models/metadata.pickle"
    metadata = load_pickle(path_metadata)
    return model, metadata


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
        else:
            df[feature] = 0
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
    scaler.transform(df[numerical_columns])
    df[numerical_columns] = scaler.transform(df[numerical_columns])
    return df
