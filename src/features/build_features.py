import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import dump_pickle


def generate_features(path: str, label: str):
    """This function allows you to:
    - generate the preprocessed dataset
    - store metadata
    - store the new dataframe

    :param path: path where you want to store the new dataset
    :type path: str
    :param label: label
    :type label: str
    """
    df, scaler, features = build_features(f"{path}data/raw/application_train.csv", label)
    metadata = {'scaler': scaler,
                'features_columns': features}
    dump_pickle(f"{path}models/metadata", metadata)
    df.to_csv(f"{path}data/processed/application_train.csv")
    print('Done')


def build_features(path: str, label: str):
    """This function allows you to:
    - build the preprocessed dataframe
    - generate the MinMax scaler
    - generate a list of the features

    :param path: path of the dataset to preprocessed
    :type path: str
    :param label: label
    :type label: str
    :return: the preprocessed dataframe
    :rtype: pd.DataFrame
    :return: the MinMax scaler
    :rtype: MinMaxScaler
    :return: a list of the features
    :rtype: list
    """
    df = pd.read_csv(path)
    df = remove_percent_missing_values(df, 35)
    number_columns = get_number_columns(df)
    df = preprocess_numerical_features(df, number_columns, label)
    df = df.drop_duplicates()
    df['cat_age'] = transform_dob_catage(df, 'DAYS_BIRTH')
    df = df.drop(columns=['DAYS_BIRTH'])
    number_columns.remove('DAYS_BIRTH')
    mode_days_employed = df[df['DAYS_EMPLOYED'] != 365243]['DAYS_EMPLOYED'].mode()[0]
    df['DAYS_EMPLOYED'] = df.DAYS_EMPLOYED.apply(lambda x: fix_anomalie_employment(x, mode_days_employed))
    df, scaler = do_min_max_scaler(df, number_columns)
    categorical_cols_train = get_columns_with_type(df, 'object')
    df = get_dummies_categorical(df, categorical_cols_train)
    features = list(df.columns)
    features.remove(label)
    df[label] = df[label].astype(int)
    return df, scaler, features


def remove_percent_missing_values(df: pd.DataFrame, percent: int):
    """This function allows you to do feature selection/reduction by deleting features with more than x% of NULL/NA values

    :param df: dataframe to modify
    :type df: pd.DataFrame
    :param percent: a limit percentage to fix in order to remove features with more than this percentage of NULL/NA values
    :type percent: int
    :return: the modified dataframe
    :rtype: pd.DataFrame
    """
    missing_values = pd.DataFrame(df.isnull().sum() * 100 / len(df), columns=['percentage'])
    to_keep = missing_values[missing_values['percentage'] < percent]
    columns_to_keep = list(to_keep.index)
    df = df[columns_to_keep]
    return df


def get_number_columns(df: pd.DataFrame):
    """This function allows you to get the number of columns in the dataframe

    :param df: the dataframe we want to get the number of columns
    :type df: pd.DataFrame
    :return: the number of columns
    :rtype: list
    """
    float_columns = get_columns_with_type(df, "float64")
    int_columns = get_columns_with_type(df, "int64")
    number_columns = list(float_columns) + list(int_columns)
    return number_columns


def preprocess_numerical_features(df: pd.DataFrame, number_columns, label: str):
    """This function allows you to fill NULL/NA values withe the mode

    :param df: the dataframe to preprocessed
    :type df: pd.DataFrame
    :param number_columns: the number of columns in the dataframe
    :type number_columns: int
    :param label: label
    :type label: str
    :return: the modified dataframe
    :rtype: pd.DataFrame
    """
    if label not in df.columns:
        number_columns.remove(label)
    for feature in number_columns:
        df[feature] = df[feature].fillna(df[feature].mode()[0]).abs()
    return df


def transform_dob_catage(df: pd.DataFrame, days_birth: str):
    """This function allows you to transform days of birth from days to year to age class

    :param df: dataframe to modify
    :type df: pd.DataFrame
    :param days_birth: column with the days of birth
    :type days_birth: str
    :return: new classe_age column
    :rtype: pd.DataFrame
    """
    temp = df.copy()
    temp['age'] = (df[days_birth]/365).astype(int)
    temp['classe_age'] = pd.cut(temp['age'], [0, 24, 35, 49, 65, 100],
                             labels=["Jeune", "Jeune_adulte", "Adulte", "Aine", "Senior"], include_lowest=True).astype('object')
    return temp['classe_age']


def fix_anomalie_employment(value, mode):
    """This function allows you to fix an anomalie by the mode

    :param value: days of employment
    :type value: str
    :param mode: mode
    :type mode: int
    :return: new value
    :rtype: int
    """
    if int(value) == 365243:
        value = int(mode)
    return value


def do_min_max_scaler(df: pd.DataFrame, number_columns):
    """This function allows you to:
     - transform features by scaling each feature to a given range
     - store the scaler

    :param df: dataframe to scale
    :type df:pd.DataFrame
    :param number_columns: the number of columns in the dataframe
    :type number_columns: int
    :return: modified dataframe
    :rtype: pd.DataFrame
    :return: scaler
    :rtype: MinMaxScaler
    """
    scaler = MinMaxScaler()
    scaler.fit(df[number_columns])
    df[number_columns] = scaler.transform(df[number_columns])
    return df, scaler


def get_columns_with_type(df: pd.DataFrame, type: str):
    """This function allows you to get all the columns with a certain type

    :param df: dataframe you want to get all the columns with a certain type
    :type df:pd.DataFrame
    :param type: the type you need
    :type type: str
    :return: dataframe with columns of this type
    :rtype: pd.DataFrame
    """
    return df.select_dtypes(include=type).columns


def get_dummies_categorical(df: pd.DataFrame, categorical_cols):
    """This function allows you to do one hot encoding on categorical columns

    :param df: dataframe you want to preprocessed
    :type df:pd.DataFrame
    :param categorical_cols: dataframe with categorical columns
    :type categorical_cols: pd.DataFrame
    :return: dataframe preprocessed
    :rtype: pd.DataFrame
    """
    return pd.get_dummies(df, columns=categorical_cols)
