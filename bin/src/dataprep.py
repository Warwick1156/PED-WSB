import pandas as pd
import ast

from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from data import load_json


def split(df: pd.DataFrame, seed: int=1156, test_size: float=0.1, split: bool=True, variant:str = 'full'):
    column = _get_column_names(variant, "column")
    data = df[[column["target"]] + column["text"] + column["numerical"]]

    return train_test_split(data, test_size=test_size, random_state=seed)


def preprocess(train: pd.DataFrame, test: pd.DataFrame, verbose: bool = True, variant: str = 'full') -> pd.DataFrame:
    settings = load_json()

    train = _quantile(train, settings)
    train, test = _minmax(train, test, variant)
    train, test = _process_tokens(train, test, settings)

    return train, test


def _get_column_names(variant, setting_column):
    settings = load_json()
    column = None
    for i in range(len(settings[setting_column])):
        if settings[setting_column][i]["name"] == variant:
            column = settings[setting_column][i]

    assert column is not None
    return column


def _quantile(df: pd.DataFrame, settings, verbose: bool = True) -> pd.DataFrame:
    quantile = settings["preprocess"][0]["quantile"][0]["quantile"]
    columns = settings["preprocess"][0]["quantile"][0]["columns"]
    original_shape = df.shape[0]

    for column in columns:
        try:
            pre_rows = df.shape[0]
            df = df[df[column] < df[column].quantile(quantile)]

            if verbose:
                print("Removed", pre_rows - df.shape[0], "rows while filtering", quantile, "quantile of", column)
        except:
            pass

    if verbose:
        print("Original rows: {}, rows after preprocessing: {}".format(original_shape, df.shape[0]))
        print("Dataframe reduced by: {:2.2%}".format(-(1. - (original_shape/df.shape[0]))))

    return df


def _minmax(train, test, variant):
    scaler = MinMaxScaler()
    columns = _get_column_names(variant, "preprocess")["minmax"]

    train[columns] = scaler.fit_transform(train[columns])
    test[columns] = scaler.transform(test[columns])

    return train, test


def _process_tokens(train, test, settings):
    train = _clear_tokens(train, settings)
    test = _clear_tokens(test, settings)

    return train, test


def _clear_tokens(df, settings):
    columns = settings["preprocess"][0]["token"]

    for column in columns:
        values = [ast.literal_eval(row) for row in df[column]]
        values = _remove_empty(values)
        values = _remove_nan(values)
        values = _to_string(values)

        df[column] = values

    return df


def _remove_empty(data):
    return [row if len(row) > 0 else [''] for row in data]


def _remove_nan(data):
    return [row if row[0] != 'nan' else [''] for row in data]


def _remove_single_character(data):
    return [[word for word in row if len(word) > 1 ] for row in data]


def _to_string(data):
    return [' '.join(row) for row in data]
