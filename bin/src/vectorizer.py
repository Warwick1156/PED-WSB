import pickle 
import os

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.data import Dataset

from data import path


def fit(data, save_name: str='', max_tokens=20000, output_sequence_length=300, output_mode='int'):
    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length, output_mode=output_mode)
    text_ds = Dataset.from_tensor_slices(data).batch(128)
    vectorizer.adapt(text_ds)
    
    if save_name != '':
        pickle.dump(
            {
                'config': vectorizer.get_config(),
                'weights': vectorizer.get_weights()
            }, open(os.path.join(path('data'), save_name), "wb"))
    
    return vectorizer
    
    
def load_vectorizer(name: str):
    pickle_file = pickle.load(open(path(name), "rb"))
    vectorizer = TextVectorization.from_config(pickle_file['config'])
    vectorizer.set_weights(pickle_file['weights'])

    return vectorizer
    