from keras import backend as K
from keras.models import load_model, Model, Sequential
from keras import layers

def build_cnn_model():
    model = Sequential()
    model.add(layers.Conv2D(filters=40, kernel_size=(5, 5), padding='valid', input_shape=(48, 48, 1)))
    model.add(layers.MaxPooling2D(pool_size=(4, 4)))
    model.add(layers.Conv2D(filters=80, kernel_size=(5, 5), padding='valid'))
    def min_max_pool2d(x):
        max_x = K.max(K.max(x, axis=1), axis=1)
        min_x = K.min(K.min(x, axis=1), axis=1)
        return K.concatenate([max_x, min_x]) # concatenate on channel

    def min_max_pool2d_output_shape(input_shape):
        return (None, input_shape[-1]*2)

    # replace maxpooling layer
    model.add(layers.Lambda(min_max_pool2d, output_shape=min_max_pool2d_output_shape))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1, activation='relu'))
    return model

def build_cnn_model2():
    model = Sequential()
    model.add(layers.Conv2D(filters=40, kernel_size=(5, 5), padding='valid', input_shape=(48, 48, 1)))
    model.add(layers.MaxPooling2D(pool_size=(4, 4)))
    model.add(layers.Conv2D(filters=80, kernel_size=(5, 5), padding='valid'))
    def min_max_pool2d(x):
        max_x = K.max(K.max(x, axis=1), axis=1)
        min_x = K.min(K.min(x, axis=1), axis=1)
        return K.concatenate([max_x, min_x]) # concatenate on channel

    def min_max_pool2d_output_shape(input_shape):
        return (None, input_shape[-1]*2)

    # replace maxpooling layer
    model.add(layers.Lambda(min_max_pool2d, output_shape=min_max_pool2d_output_shape))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1))
    return model
