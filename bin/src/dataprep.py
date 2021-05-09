import pandas as pd
import ast

from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from data import load_json


def split(df: pd.DataFrame, seed: int=1156, test_size: float=0.1, split: bool=True):
    settings = load_json()
    column = settings["column"][0]
    
    data = df[[column["target"]] + column["text"] + column["numerical"]]
    
    return train_test_split(data, test_size=test_size, random_state=seed)
    
    
def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    settings = load_json()
    
    train, test = _minmax(train, test, settings)
    train, test = _process_tokens(train, test, settings)
#    train, test = _join_tokens()
    
    return train, test
        
        
def _minmax(train, test, settings):
    scaler = MinMaxScaler()
    columns = settings["preprocess"][0]["minmax"]
    
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
 #       values = _remove_single_character(values)
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
