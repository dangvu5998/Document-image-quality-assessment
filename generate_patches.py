import argparse
import os
import cv2
from .utils import generate_patches, get_document_corners, four_point_transform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', help='File contains images path and score')
    parser.add_argument('--dist', help='Destination directory for data')
    parser.add_argument('--width_patch', help='Width of patches')
    parser.add_argument('--height_patch', help='Height of patches')
    parser.add_argument('--data_path', help='Data directory path')
    args = parser.parse_args()
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
        corners = get_document_corners(img, img_type='BGR')
        img = four_point_transform(img, corners)
        patches = generate_patches(img, img_type='BGR', blank_threshold=0.9)
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