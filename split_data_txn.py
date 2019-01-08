import argparse
import os
import random

def main():
    '''
    Read label file and split label file to 3 file training, testing, validation
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist_path', help='Destination directory for split')
    parser.add_argument('--data_path', help='Data directory path')
    parser.add_argument('--seed', help='Seed for random shuffle')
    parser.add_argument('--test_scale', help='Test scale in split')
    parser.add_argument('--val_scale', help='Val scale in split')
    args = parser.parse_args()
    seed = args.seed
    if seed is not None:
        random.seed(int(seed))
    else:
        random.seed(101)
    data_path = args.data_path
    dist_path = args.dist_path
    test_scale = float(args.test_scale) if args.test_scale is not None else 0.2
    val_scale = float(args.val_scale) if args.val_scale is not None else 0.2
    labels_fp = open(os.path.join(data_path, 'labels.txt'), encoding='utf-8')
    labels = []
    for line in labels_fp:
        line = line.strip()
        if len(line) > 0:
            labels.append(line)
    labels_fp.close()
    nb_of_labels = len(labels)
    start_val_index = int((1 - val_scale - test_scale)*nb_of_labels)
    start_test_index = int(start_val_index + test_scale*nb_of_labels)
    train_label_f = open(os.path.join(dist_path, 'train_labels.txt'), 'w')
    random.shuffle(labels)
    for label in labels[:start_val_index]:
        train_label_f.write(label+'\n')
    train_label_f.close()
    val_label_f = open(os.path.join(dist_path, 'val_labels.txt'), 'w')
    for label in labels[start_val_index:start_test_index]:
        val_label_f.write(label+'\n')
    val_label_f.close()
    test_label_f = open(os.path.join(dist_path, 'test_labels.txt'), 'w')
    for label in labels[start_test_index:]:
        test_label_f.write(label+'\n')
    test_label_f.close()

if __name__ == '__main__':
    main()
