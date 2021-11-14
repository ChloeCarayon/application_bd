import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def build_features(train_path: str, test_path: str, label: str):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_train, df_test = remove_percent_missing_values(df_train, df_test, label, 35)
    number_columns = get_number_columns(df_train)
    df_train = preprocess_numerical_features(df_train, number_columns, label)
    df_test = preprocess_numerical_features(df_test, number_columns, label)
    df_train = df_train.drop_duplicates()
    df_train['cat_age'] = transform_dob_catage(df_train, 'DAYS_BIRTH')
    df_test['cat_age'] = transform_dob_catage(df_test, 'DAYS_BIRTH')
    df_train = df_train.drop(columns=['DAYS_BIRTH'])
    df_test = df_test.drop(columns=['DAYS_BIRTH'])
    number_columns.remove('DAYS_BIRTH')
    mode_days_employed = df_train[df_train['DAYS_EMPLOYED'] != 365243]['DAYS_EMPLOYED'].mode()[0]
    df_train['DAYS_EMPLOYED'] = df_train.DAYS_EMPLOYED.apply(lambda x: fix_anomalie_employment(x, mode_days_employed))
    df_test['DAYS_EMPLOYED'] = df_test.DAYS_EMPLOYED.apply(lambda x: fix_anomalie_employment(x, mode_days_employed))
    do_min_max_scaler(df_train, df_test, number_columns)
    categorical_cols_train = get_categorical_columns(df_train)
    df_train = get_dummies_categorical(df_train, categorical_cols_train)
    categorical_cols_test = get_categorical_columns(df_test)
    df_test = get_dummies_categorical(df_test, categorical_cols_test)
    df_test = drop(df_train, df_test)


def remove_percent_missing_values(df_train: pd.DataFrame, df_test: pd.DataFrame, label: str, percent: int):
    missing_values = pd.DataFrame(df_train.isnull().sum() * 100 / len(df_train), columns=['percentage'])
    to_keep = missing_values[missing_values['percentage'] < percent]
    columns_to_keep = list(to_keep.index)
    df_train = df_train[columns_to_keep]
    columns_to_keep.remove(label)
    df_test = df_test[columns_to_keep]
    columns_to_keep.append(label)
    return df_train, df_test


def get_number_columns(df: pd.DataFrame):
    return list(df.index)


def preprocess_numerical_features(df: pd.DataFrame, number_columns, label: str):
    if label not in df.columns:
        number_columns.remove(label)
    for feature in number_columns:
        df[feature].fillna(df[feature].mode()[0], inplace=True)
        df[feature] = df[feature].abs()
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


def do_min_max_scaler(df_train: pd.DataFrame, df_test: pd.DataFrame, number_columns):
    scaler = MinMaxScaler()
    scaler.fit(df_train[number_columns])
    df_train[number_columns] = scaler.transform(df_train[number_columns])
    df_test[number_columns] = scaler.transform(df_test[number_columns])


def get_categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(include='object').columns


def get_dummies_categorical(df: pd.DataFrame, categorical_cols):
    return pd.get_dummies(df, columns=categorical_cols)


def drop(df_train: pd.DataFrame, df_test: pd.DataFrame):
    for col in df_train.columns:
        if col not in df_test.columns:
            df_test[col] = 0

    columns_drop = []
    for col in df_test.columns:
        if col not in list(df_train.columns):
            columns_drop.append(col)
    return df_test.drop(columns=columns_drop)
