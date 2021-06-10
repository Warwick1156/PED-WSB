import os
import ast
import sys
src_dir = os.path.join('..', 'src')
sys.path.append(os.path.abspath(src_dir))

import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

from dataprep import split, preprocess
from data import path, get_dataset
from embbeding import get_embbeding_layer
from vectorizer import load_vectorizer


def dataframe_to_dataset(dataframe, weight:bool = False):
    dataframe = dataframe.copy()
    if weight:
        sample_weight = weight_samples(dataframe)
    else:
        sample_weight = np.ones(dataframe.shape[0])

    labels = dataframe.pop("score")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels, sample_weight))

    return ds


def encode_text_feature(feature, name, dataset):
    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=100, output_mode='int')

    feature_ds = dataset.map(lambda x, y, z: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    vectorizer.adapt(feature_ds)
    encoded_feature = vectorizer(feature)

    return encoded_feature


def embbed_vectorized_text(feature):
    return layers.Embedding(20000, 64)(feature)


def add_lstm(feature, out_dim=16, return_sequences=False):
    # return layers.Bidirectional(layers.LSTM(out_dim, return_sequences=return_sequences))(feature)
    return layers.LSTM(out_dim, return_sequences=return_sequences)(feature)


def text_input(columns):
    return [keras.Input(shape=(1,), name=column, dtype=tf.string) for column in columns]


def numerical_input(columns):
    return [keras.Input(shape=(1,), name=column, dtype="float64") for column in columns]


def vectorize_inputs(input_layers, columns, dataset):
    result = []
    for i, layer in enumerate(input_layers):
        result.append(encode_text_feature(layer, columns[i], dataset))

    return result


def embedding_layers(in_layers):
    return [embbed_vectorized_text(layer) for layer in in_layers]


def add_lstm_layers(in_layers, out_dim=2, return_sequences=False):
    return [add_lstm(layer, out_dim, return_sequences) for layer in in_layers]


def get_model(train, train_ds):
    text_columns = train.iloc[:, 1:5].columns
    # text_columns.append('title_stem_tokens')
    numerical_columns = train.iloc[:, 5:].columns

    text_inputs = text_input(text_columns)
    numerical_inputs = numerical_input(numerical_columns)
    all_inputs = text_inputs + numerical_inputs

    vectorized_text_layers = vectorize_inputs(text_inputs, text_columns, train_ds)
    embedded_text_layers = embedding_layers(vectorized_text_layers)

    lstm = add_lstm_layers(embedded_text_layers, 32, True)
    lstm = add_lstm_layers(lstm, 16)

    text = layers.concatenate(lstm)

    numerical = layers.concatenate(numerical_inputs)

    # y = layers.LSTM(128, return_sequences=True)(text)
    # y = layers.LSTM(32)(y)

    y = layers.Flatten()(text)

    # x = layers.Dense(128, activation="relu")(numerical)
    # x = layers.Dropout(0.5)(x)

    # all_ = layers.concatenate([y, x])
    all_ = layers.concatenate([y, numerical])
    z = layers.Dense(64, activation="relu")(all_)
    z = layers.Dropout(0.5)(z)
    z = layers.Dense(32, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(16, activation="relu")(z)

    output = layers.Dense(1, activation="linear")(z)
    model = keras.Model(all_inputs, output)
    model.compile("sgd", loss="mean_absolute_error")
    keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

    return model


def get_callbacks():
    filepath = os.path.join(path('checkpoint'), 'model.{epoch:02d}-{val_loss:.2f}.ckpt')
    return [
        tf.keras.callbacks.EarlyStopping(patience=50),
        tf.keras.callbacks.ModelCheckpoint(filepath=filepath, save_weights_only=True, save_best_only=True, verbose=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]


def weight_samples(df: pd.DataFrame):
    tqdm.pandas()
    print("Weighting samples...")

    weight = []
    df.score.progress_apply(lambda x: weight.append(df[df.score < x].shape[0] / df.shape[0]))

    return np.array(weight)


if __name__ == '__main__':
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    data = get_dataset()
    data.is_oc = data.is_oc.astype(np.float32)
    data.is_self = data.is_self.astype(np.float32)

    # data = data[~(
    #         (data.created > 1611702000)
    #         & (data.created < 1611874799)
    # )]

    train, test = split(data, variant='relevant')
    train, test = preprocess(train, test, variant='relevant')

    train.drop(columns=['image_label', 'body_stem_tokens'], inplace=True)
    test.drop(columns=['image_label', 'body_stem_tokens'], inplace=True)

    train_ds = dataframe_to_dataset(train, weight=True)
    validation_ds = dataframe_to_dataset(test)

    train_ds = train_ds.batch(32)
    validation_ds = validation_ds.batch(32)

    model = get_model(train, train_ds)
    model.fit(train_ds, epochs=500, callbacks=get_callbacks(), validation_data=validation_ds,)

    y = model.predict(validation_ds)
    y.tolist()
    y = [int(x[0]) for x in y]
    plt.figure(figsize=(20, 20))
    sns.scatterplot(x=test.score.values, y=y)
    plt.ylim([0, 2000])
    plt.xlim([0, 2000])
    plt.show()
