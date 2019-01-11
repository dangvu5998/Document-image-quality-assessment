import os
import argparse
import logging
import tensorflow as tf
from sklearn.utils import shuffle
from .models import cnn_model_fn

def data_input_fn(image_paths, labels, batch_size):
    image_paths = tf.constant(image_paths)
    labels = tf.constant(labels, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    def _parse_image(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_decoded = tf.to_float(image_decoded)
        image = tf.multiply(tf.subtract(image_decoded, 128), 1/128)
        return image_decoded, label

    return dataset.shuffle(50000).map(_parse_image).batch(batch_size)

def read_metadata_dataset(data_path, is_shuffle=True):
    image_paths = []
    labels = []
    with open(os.path.join(data_path, 'labels.txt'), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_fn, label = line.split('\t')
            label = float(label)
            image_path = os.path.join(data_path, 'images', image_fn)
            image_paths.append(image_path)
            labels.append(label)
    if is_shuffle:
        return shuffle(image_paths, labels)
    else:
        return image_paths, labels

def main():
    '''Training estimator'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size for data generator')
    parser.add_argument('--training_path', help='dataset for training')
    parser.add_argument('--epochs', help='Number of epochs to train the model')
    parser.add_argument('--validation_path', help='dataset for validation')
    parser.add_argument('--checkpoint_dir', help='model checkpoint path')
    # parser.add_argument('--log_path', help='log info while training')
    args = parser.parse_args()
    batch_size = int(args.batch_size) if args.batch_size else 32
    epochs = int(args.epochs) if args.epochs else 5
    checkpoint_dir = args.checkpoint_dir
    training_path = args.training_path
    validation_path = args.validation_path
    tf.logging.set_verbosity(tf.logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # log = logging.getLogger('tensorflow')
    # file_logger = logging.FileHandler(args.log_path)
    # file_logger.setFormatter(formatter)
    # log.addHandler(file_logger)
    train_image_paths, train_labels = read_metadata_dataset(training_path)
    val_image_paths, val_labels = read_metadata_dataset(validation_path)
    # train_image_paths = train_image_paths[:100]
    # train_labels = train_labels[:100]
    # val_image_paths = val_image_paths[:100]
    # val_labels = val_labels[:100]
    classifier = tf.estimator.Estimator(cnn_model_fn, model_dir=checkpoint_dir)

    for epoch in range(epochs):
        classifier.train(
            input_fn=lambda: data_input_fn(train_image_paths, train_labels, batch_size)
        )
        evaluation = classifier.evaluate(
            input_fn=lambda: data_input_fn(val_image_paths, val_labels, batch_size),
        )
        print('--------------------------------------------------------------------')
        print('Epoch:', epoch+1, '\nEvaluation:', evaluation)

if __name__ == '__main__':
    main()
