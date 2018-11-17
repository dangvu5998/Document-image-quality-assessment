import os
import argparse
from random import shuffle
import numpy as np
import imageio
import pandas as pd
import keras as K
from keras.models import load_model, Model, Sequential
from keras import layers, optimizers, callbacks
import tensorflow as tf
from .utils import image_normalize

dir_path = os.path.dirname(os.path.realpath(__file__))
pretrained_directory = os.path.join(dir_path, 'pretrained')
model_path = None

def build_model():
    model = Sequential()
    model.add(layers.Conv2D(filters=40, kernel_size=(5, 5), padding='valid', input_shape=(48, 48, 1)))
    model.add(layers.MaxPooling2D(pool_size=(4, 4)))
    model.add(layers.Conv2D(filters=80, kernel_size=(5, 5), padding='valid'))
    def min_max_pool2d(x):
        max_x = K.backend.max(K.backend.max(x, axis=1), axis=1)
        min_x = K.backend.min(K.backend.min(x, axis=1), axis=1)
        return K.backend.concatenate([max_x, min_x]) # concatenate on channel

    def min_max_pool2d_output_shape(input_shape):
        return (None, input_shape[-1]*2)

    # replace maxpooling layer
    model.add(layers.Lambda(min_max_pool2d, output_shape=min_max_pool2d_output_shape))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1, activation='relu'))
    return model

class DataGenerator:
    def __init__(self, data_path, batch_size=32):
        self.data_path = data_path
        labels_file = open(os.path.join(data_path, 'labels.txt'), 'r')
        # skip the first line
        labels_file.readline()
        self.batch_size = batch_size
        self.lines = labels_file.readlines()
        shuffle(self.lines)
        self.nb_batch = len(self.lines)//batch_size
        self.curr_batch = self.nb_batch

    def __next__(self):
        if self.curr_batch >= self.nb_batch:
            shuffle(self.lines)
            self.curr_batch = 0
        images = []
        labels = []
        for i in range(self.curr_batch*self.batch_size, (self.curr_batch+1)*self.batch_size):
            img_filename, score = self.lines[i].split('\t')
            img_path = os.path.join(self.data_path, 'img', img_filename)
            score = float(score)
            image = imageio.imread(img_path)
            image = image_normalize(image)
            images.append(image)
            labels.append([score])
        self.curr_batch += 1
        return np.asarray(images), np.asarray(labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size for data generator')
    parser.add_argument('--lr', help='learning rate for training')
    parser.add_argument('--training_path', help='dataset for training')
    parser.add_argument('--epochs', help='Number of epochs to train the model')

    args = parser.parse_args()
    lr = args.lr or 0.0001
    batch_size = args.batch_size or 32
    training_path = args.training_path or './DIQA_training'
    epochs = args.epochs or 5
    optimizer = optimizers.Adam(lr=lr)
    training_generator = DataGenerator(training_path, batch_size)
    model = build_model()
    print(model.summary())
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])
    checkpoint_path = './checkpoints/cnn-{epoch:02d}-{loss:.4f}.h5'
    checkpoint_cb = callbacks.ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True)
    cb_list = [checkpoint_cb]
    model.fit_generator(training_generator, steps_per_epoch=training_generator.nb_batch, epochs=epochs, callbacks=cb_list)

if __name__ == '__main__':
    main()