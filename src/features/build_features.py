import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import dump_pickle


def generate_features(path: str, label: str):
    df, scaler, features = build_features(f"{path}/train/application_train.csv", label)
    metadata = {'scaler': scaler,
                'features_columns': features}
    dump_pickle(f"{path}/train/metadata", metadata)
    df.to_csv(f"{path}/train/preprocessed_train.csv")


def build_features(train_path: str, label: str):
    df = pd.read_csv(train_path)
    df = remove_percent_missing_values(df, label, 35)
    number_columns = get_number_columns(df)
    df = preprocess_numerical_features(df, number_columns, label)
    df = df.drop_duplicates()
    df['cat_age'] = transform_dob_catage(df, 'DAYS_BIRTH')
    df = df.drop(columns=['DAYS_BIRTH'])
    number_columns.remove('DAYS_BIRTH')
    mode_days_employed = df[df['DAYS_EMPLOYED'] != 365243]['DAYS_EMPLOYED'].mode()[0]
    df['DAYS_EMPLOYED'] = df.DAYS_EMPLOYED.apply(lambda x: fix_anomalie_employment(x, mode_days_employed))
    df, scaler = do_min_max_scaler(df, number_columns)
    categorical_cols_train = get_type_columns(df, 'object')
    df = get_dummies_categorical(df, categorical_cols_train)
    features = list(df.columns)
    features.remove(label)
    return df, scaler, features


def remove_percent_missing_values(df: pd.DataFrame, label: str, percent: int):
    missing_values = pd.DataFrame(df.isnull().sum() * 100 / len(df), columns=['percentage'])
    to_keep = missing_values[missing_values['percentage'] < percent]
    columns_to_keep = list(to_keep.index)
    df = df[columns_to_keep]
    return df


def get_number_columns(df: pd.DataFrame):
    float_columns = get_type_columns(df, "float64")
    int_columns = get_type_columns(df, "int64")
    number_columns = list(float_columns) + list(int_columns)
    return number_columns


def preprocess_numerical_features(df: pd.DataFrame, number_columns, label: str):
    if label not in df.columns:
        number_columns.remove(label)
    for feature in number_columns:
        df[feature] = df[feature].fillna(df[feature].mode()[0]).abs()
    return df


def transform_dob_catage(df: pd.DataFrame, days_birth: str):
    temp = df.copy()
    temp['age'] = (df[days_birth]/365).astype(int)
    temp['cat_age'] = pd.cut(temp['age'], [0, 24, 35, 49, 65, 100],
                             labels=[0, 1, 2, 3, 4], include_lowest=True).astype('object')
    return temp['cat_age']


def fix_anomalie_employment(value, mode):
    if int(value) == 365243:
        value = mode
    return value


def do_min_max_scaler(df: pd.DataFrame, number_columns):
    scaler = MinMaxScaler()
    scaler.fit(df[number_columns])
    df[number_columns] = scaler.transform(df[number_columns])
    return df, scaler


def get_type_columns(df: pd.DataFrame, type: str):
    return df.select_dtypes(include=type).columns


def get_dummies_categorical(df: pd.DataFrame, categorical_cols):
    return pd.get_dummies(df, columns=categorical_cols)
