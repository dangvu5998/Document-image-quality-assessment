import os
import argparse
from random import shuffle
import numpy as np
import imageio
import pandas as pd
from keras import backend as K
from keras.models import load_model, Model, Sequential
from keras import optimizers, callbacks
from .utils import image_normalize
from .models import build_cnn_model, build_cnn_model2

dir_path = os.path.dirname(os.path.realpath(__file__))
pretrained_directory = os.path.join(dir_path, 'pretrained')
model_path = None

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
    parser.add_argument('--validation_path', help='dataset for validation')
    parser.add_argument('--checkpoint_directory', help='model checkpoint path')
    parser.add_argument('--save_best_only', help='only save best model checkpoint path')
    args = parser.parse_args()
    lr = float(args.lr) if args.lr else 0.0001
    batch_size = int(args.batch_size) if args.batch_size else 32
    training_path = args.training_path or './DIQA_training'
    epochs = int(args.epochs) if args.epochs else 5
    optimizer = optimizers.Adam(lr=lr)
    training_generator = DataGenerator(training_path, batch_size)
    validation_generator = None
    validation_steps = None
    if args.validation_path:
        validation_generator = DataGenerator(args.validation_path, batch_size)
        validation_steps = validation_generator.nb_batch
    model = build_cnn_model2()
    print(model.summary())
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    checkpoint_directory = args.checkpoint_directory or  './checkpoints'
    checkpoint_path = os.path.join(checkpoint_directory, 'cnn1-{epoch:02d}-mae-{val_loss:.3f}.h5')
    save_best_only = args.save_best_only
    if not save_best_only or save_best_only == 'false':
        save_best_only = False
    else:
        save_best_only = True
    checkpoint_cb = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=save_best_only)
    cb_list = [checkpoint_cb]
    model.fit_generator(training_generator,
                        steps_per_epoch=training_generator.nb_batch, 
                        epochs=epochs,
                        validation_steps=validation_steps,
                        validation_data=validation_generator,
                        callbacks=cb_list)

if __name__ == '__main__':
    main()