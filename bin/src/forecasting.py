import numpy as np
import pandas as pd

from tqdm import tqdm
from tensorflow import keras
from matplotlib import pyplot as plt
from math import ceil
from pathlib import Path

from data import get_dataset, path
from models import lstm1


def train_test_split(df, test_size: float = 0.1, random_state: int = 130759):
    ids_df = pd.DataFrame(df.post_id.unique())
    test_ids = ids_df.sample(int(ids_df.shape[0] * test_size), random_state=random_state)

    mask = df.post_id.isin(test_ids.iloc[:, 0])

    return df[~mask], df[mask]


def reshape_for_lstm(df, x_columns: list = ['new_comments'], y_column: str = 'cumsum', samples_per_post: int = 288, past: int = 6, future: int = 12):
    assert df.shape[0] % samples_per_post == 0, 'Data shape not dividable by samples_per_post number'

    x_list = []
    y_list = []
    for i in tqdm(range(0, df.shape[0], samples_per_post)):
        subset = df.iloc[i:i + samples_per_post, :]
        subset.reset_index(inplace=True)

        x = subset.iloc[:past, :].loc[:, x_columns].to_numpy().reshape((past, len(x_columns)))
        y = subset.iloc[future, subset.columns.get_loc(y_column)]

        x_list.append(x)
        y_list.append(y)

    return np.array(x_list), np.array(y_list)


def plot(plot_data, future, y_pred, y_true, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    #     time_steps = list(range(-(plot_data[0].shape[0]), 0))
    time_steps = [-x for x in range(len(plot_data), 0, -1)]

    plt.title(title)
    plt.plot(time_steps, plot_data, marker[0], markersize=10, label=labels[0])
    plt.plot([future], [y_pred], marker[2], label=labels[2])
    plt.plot([future], [y_true], marker[1], label=labels[1])

    plt.legend()
    plt.xlim([time_steps[0], (future + 5)])
    plt.xlabel("Time-Step")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def get_random_ids(df, samples: int, random_state: int):
    return pd.DataFrame(df.post_id.unique()).sample(samples, random_state=random_state).iloc[:, 0].values


def make(make_model, past, future, epochs, n_charts, seed, name, verbose):
    data = get_dataset('timeseries_v4.1.csv')
    train, test = train_test_split(data)
    x_train, y_train = reshape_for_lstm(train, ['new_comments'], 'cumsum', past=past, future=future)
    x_test, y_test = reshape_for_lstm(train, ['new_comments'], 'cumsum', past=past, future=future)

    model = make_model((x_train.shape[1], x_train.shape[2]))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mae")
    print(model.summary())

    path_checkpoint = Path(path('model'), name + '.h5')
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=25)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=epochs,
        callbacks=[es_callback, modelckpt_callback],
        validation_split=0.2,
        verbose=verbose,
    )

    print(f'MAE on test set: {pd.DataFrame(keras.losses.MAE(y_test, model.predict(x_test))).mean()}')

    test_ids = get_random_ids(test, n_charts, seed)
    print(f'Test ids: {test_ids}')

    for id_ in test_ids:
        subset = test[test.post_id == id_]
        x, y = reshape_for_lstm(subset, past=past, future=future)
        x_axis = subset['cumsum'].values[:30]
        y_pred = ceil(model.predict(x)[0][0])
        plot(x_axis, future, y_pred, y, id_)


if __name__ == '__main__':
    # data = get_dataset('timeseries_v4.1.csv')
    # train, test = train_test_split(data)
    # x_train, y_train = reshape_for_lstm(train, ['new_comments'], 'cumsum', future=6*24)
    # x_test, y_test = reshape_for_lstm(train, ['new_comments'], 'cumsum', future=6 * 24)
    #
    # model = lstm1((x_train.shape[1], x_train.shape[2]))
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    # print(model.summary())
    #
    # history = model.fit(
    #     x_train,
    #     y_train,
    #     batch_size=32,
    #     epochs=10,
    #     # callbacks=callbacks,
    #     validation_split=0.2,
    #     verbose=1,
    # )
    #
    # # print(model.predict(x_test[50:100]), y_test[50:100])
    # # print(keras.losses.MAE(y_test, model.predict(x_test)))
    #
    # for x, y in zip(x_test, y_test)[:5]:
    #     x_list = [x[i][0] for i in range(len(x))]
    #     y_pred = model.predict()

    make(lstm1, past=6 * 1, future=6 * 6, epochs=100, n_charts=5, seed=1156, name='lstm1_1_6', verbose=0)

