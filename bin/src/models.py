from tensorflow import keras


def lstm1(shape: tuple):
    input_layer = keras.layers.Input(shape)
    x = keras.layers.LSTM(32, activation='relu')(input_layer)

    output_layer = keras.layers.Dense(1, activation='linear')(x)
    return keras.Model(inputs=input_layer, outputs=output_layer)


def lstm2(shape: tuple):
    input_layer = keras.layers.Input(shape)
    x = keras.layers.LSTM(64, activation='relu', return_sequences=True)(input_layer)
    x = keras.layers.LSTM(32, activation='relu')(x)

    output_layer = keras.layers.Dense(1, activation='linear')(x)
    return keras.Model(inputs=input_layer, outputs=output_layer)


def lstm3(shape: tuple):
    input_layer = keras.layers.Input(shape)
    x = keras.layers.LSTM(32, activation='relu', return_sequences=True)(input_layer)
    x = keras.layers.LSTM(16, activation='relu')(x)
    x = keras.layers.Dense(8, activation='relu')(x)

    output_layer = keras.layers.Dense(1, activation='linear')(x)
    return keras.Model(inputs=input_layer, outputs=output_layer)