import numpy as np

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.data import Dataset
from keras import initializers

from data import path, load_json


class EmbbeddingNotFound(Exception):
    pass

    
    
def _get_word_index(vocabulary, verbose=True):
#    voc = [str(x)[2:-1] for x in vocabulary]
    voc = vocabulary
    
    if verbose:
        print('Smaple of vocabulary:', voc[2:7], '...')

    return dict(zip(voc, range(len(voc))))
    
    
def _get_embbedings_index(embedding_path: str, verbose: bool=True):

    embeddings_index = {}
    with open(embedding_path, encoding='utf-8') as f:   
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs


    if verbose:
        print("Found %s word vectors." % len(embeddings_index))
        
    return embeddings_index
     
        
def _get_matrix(num_tokens, embedding_dim, word_index, embeddings_index, verbose=True):
    hits = 0
    misses = []
    
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses.append(word)

    if verbose:
        print("Converted %d words (%d misses)" % (hits, len(misses)))
        
    return embedding_matrix
     
     
def _get_embbeding_dim(settings, target):
    for i in range(len(settings)):
        if settings[i]["name"] == target:
            return settings[i]["dim"]
            
    raise EmbbeddingNotFound("Settings for embbeding not found")
    
    
def get_embbeding_layer(target_embbeding: str, vocabulary, verbose=True):
    settings = load_json()["embbeding"]
    embedding_dim = _get_embbeding_dim(settings, target_embbeding)
    embedding_path = path(target_embbeding)
    
    word_index = _get_word_index(vocabulary, verbose)
    embeddings_index = _get_embbedings_index(embedding_path, verbose)
    
    num_tokens = len(vocabulary) + 2
    
    embbeding_matrix = _get_matrix(num_tokens, embedding_dim, word_index, embeddings_index, verbose)
    
    return Embedding(num_tokens, embedding_dim, embeddings_initializer=initializers.Constant(embbeding_matrix), trainable=False,), embbeding_matrix
    
    