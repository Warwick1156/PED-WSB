import os
import sys
import json
from pprint import pprint

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

src_dir = os.path.join('..', 'src')
sys.path.append(os.path.abspath(src_dir))

import pandas as pd
from tqdm import tqdm

from data import path, get_dataset
from multiprocessing_ import parallelize_dataframe


def get_cumulative(df):
    result_df = pd.DataFrame()

    for id_ in tqdm(df.post_id.unique()):
        subset = df[df.post_id == id_]
        subset.loc[:, 'cumsum'] = subset.new_comments.cumsum()

        result_df = result_df.append(subset)

    return result_df

if __name__ == '__main__':

    data = get_dataset('timeseries_v4.1.csv')
    data = data[data.post_id == 'l24zkw']
    data = get_cumulative(data)

    x_train = data.loc[:, ['new_comments']].values
    y_train = data.loc[:, 'cumsum']

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=6,
        sampling_rate=1,
        batch_size=1,
    )

    for batch in dataset_train.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)

    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    history = model.fit(
        dataset_train,
        epochs=10,
    #     validation_data=dataset_val,
    #     callbacks=[es_callback, modelckpt_callback],
    )