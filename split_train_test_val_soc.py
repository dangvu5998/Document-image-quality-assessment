import argparse
import os
import numpy as np
from .utils import get_datasets, get_score_from_eval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist_path', help='Destination directory for split')
    parser.add_argument('--data_path', help='Data directory path')
    parser.add_argument('--seed', help='Seed for random shuffle')
    parser.add_argument('--test_scale', help='Test scale in split')
    parser.add_argument('--val_scale', help='Val scale in split')
    parser.add_argument('--random', help='Random split')
    args = parser.parse_args()
    seed = args.seed
    if seed is not None:
        np.random.seed(int(seed))
    data_path = args.data_path
    dist_path = args.dist_path
    test_scale = float(args.test_scale) if args.test_scale is not None else 0.2
    val_scale = float(args.val_scale) if args.val_scale is not None else 0.2
    img_paths, eval_paths = get_datasets(
        os.path.join(data_path, 'DIQA_Release_1.0_Part1'),
        os.path.join(data_path, 'DIQA_Release_1.0_Part2')
    )
    img_paths = [img_path[len(data_path)+1:] for img_path in img_paths]
    indices = np.arange(len(img_paths))
    np.random.shuffle(indices)
    test_f = open(os.path.join(dist_path, 'test_data.txt'), 'w')
    end_test = int(test_scale*len(indices))
    for i in range(end_test):
        test_f.write('{}\t{}\n'.format(img_paths[indices[i]], get_score_from_eval(eval_paths[indices[i]])))
    test_f.close()

    end_val = end_test + int(val_scale*len(indices))
    val_f = open(os.path.join(dist_path, 'val_data.txt'), 'w')
    for i in range(end_test, end_val):
        val_f.write('{}\t{}\n'.format(img_paths[indices[i]], get_score_from_eval(eval_paths[indices[i]])))
    val_f.close()
    train_f = open(os.path.join(dist_path, 'train_data.txt'), 'w')
    for i in range(end_val, len(indices)):
        train_f.write('{}\t{}\n'.format(img_paths[indices[i]], get_score_from_eval(eval_paths[indices[i]])))
    train_f.close()

if __name__ == '__main__':
    main()
