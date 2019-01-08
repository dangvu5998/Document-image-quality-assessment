import argparse
import os
import cv2
from .utils import generate_patches, get_document_corners, four_point_transform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', help='File contains images path and score')
    parser.add_argument('--dist', help='Destination directory for data')
    parser.add_argument('--data_path', help='Data directory path')
    parser.add_argument('--blank_threshold', help='Threshold blank of patches')
    args = parser.parse_args()
    blank_threshold = args.blank_threshold or 0.95
    blank_threshold = float(blank_threshold)
    dist = args.dist
    data_path = args.data_path
    img_directory_path = os.path.join(dist, 'img')
    os.makedirs(img_directory_path)
    labels_file = open(os.path.join(dist, 'labels.txt'), 'w')
    metadata_file = open(args.metadata_path)
    index = 0
    for line in metadata_file:
        line = line.strip()
        if len(line) == 0:
            break
        img_path, score = line.split('\t')
        img_path = os.path.join(data_path, img_path)
        print('Generating: ', img_path)
        img = cv2.imread(img_path)
        patches = generate_patches(img, 
            img_type='BGR',
            blank_threshold=blank_threshold,
            document_crop=False)
        for patch in patches:
            filename = 'img_{}.jpg'.format(index)
            cv2.imwrite(
                os.path.join(img_directory_path, filename),
                patch)
            labels_file.write('{}\t{}\n'.format(filename, score))
            index += 1
        print('Generate {} patches'.format(len(patches)))
    labels_file.close()
    metadata_file.close()

if __name__ == '__main__':
    main()