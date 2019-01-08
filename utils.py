'''
Utility for image processing
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist

def order_points(pts):
    '''
    Sort four point in order top left, top right, bottom left, bottom right
    '''
    # sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (top_left, bottom_left) = left_most

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    distance = dist.cdist(top_left[np.newaxis], right_most, "euclidean")[0]
    (bottom_right, top_right) = right_most[np.argsort(distance)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

def four_point_transform(image, corners):
    '''
    Crop image with four corners
    '''
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(corners)
    (top_left, top_right, bottom_right, bottom_left) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) +\
        ((bottom_right[1] - bottom_left[1]) ** 2))
    width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) +\
        ((top_right[1] - bottom_right[1]) ** 2))
    height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) +\
        ((top_left[1] - bottom_left[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

    # return the warped image
    return warped

def is_rectangle(verticles):
    '''
    Check polygone is rectangle
    '''
    if len(verticles) != 4:
        return False
    for i in range(-1, 3):
        verticle1 = verticles[i - 1] - verticles[i]
        verticle2 = verticles[i + 1] - verticles[i]
        d_cosin = dist.cosine(verticle1, verticle2)
        if d_cosin < 0.75:
            return False
    return True

def get_document_corners(img, img_type='BGR'):
    '''
    Get corners of document image
    '''
    border_img = 40
    if img_type == 'BGR':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img_type == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    origin_h, origin_w = img.shape
    scale = origin_w/600
    new_h = int(origin_h/scale)
    new_w = 600
    img = cv2.resize(img, (new_w, new_h))
    corners_img_std = np.std(img)
    img = cv2.copyMakeBorder(img, border_img, border_img, border_img,\
                             border_img, cv2.BORDER_CONSTANT)
    img = cv2.medianBlur(img, 11)
    edges = cv2.Canny(img, 10, 30)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1) # increase connect edges
    _, cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    simplified_cnts = []
    for cnt in cnts:
        hull = cv2.convexHull(cnt)
        simplified_cnts.append(cv2.approxPolyDP(hull, 0.05*cv2.arcLength(hull, True), True))
    simplified_cnts = sorted(simplified_cnts, key=cv2.contourArea, reverse=True)
    area_th = cv2.contourArea(simplified_cnts[0])/5
    corners = None
    for i in range(1, min(5, len(simplified_cnts))):
        cnt = simplified_cnts[i].reshape(-1, 2)
        if cv2.contourArea(cnt) < area_th:
            break
        if is_rectangle(cnt):
            current_img = four_point_transform(img, cnt)
            current_img_std = np.std(current_img)
            if corners is None or current_img_std*1.2 < corners_img_std:
                corners = cnt
                corners_img_std = current_img_std
                # break
    if corners is None:
        corners = simplified_cnts[0].reshape(-1, 2)
    # corners = simplified_cnts[2]
    corners -= border_img
    corners = (corners.reshape(-1, 2)*scale).astype(np.int32)
    return corners

def generate_patches(img, patch_size=(48, 48), img_type='RGB', blank_threshold=1.0,
                     row_step=None, col_step=None, document_crop=True):
    '''
    Generate patches for training
    '''
    if document_crop:
        corners = get_document_corners(img, img_type)
        img = four_point_transform(img, corners)
    if img_type == 'BGR':
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img_type == 'RGB':
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img_type == 'GRAY':
        gray_img = img
    document_size = (1800, 2500)
    gray_img = cv2.resize(gray_img, document_size)
    blur_img = cv2.medianBlur(gray_img, 15)
    thres = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    patches = []
    rows, cols = gray_img.shape
    if row_step is None:
        row_step = int(patch_size[1]/3)
    if col_step is None:
        col_step = int(patch_size[0]/3)
    for i in range(0, rows-patch_size[1], row_step):
        for j in range(0, cols-patch_size[0], col_step):
            if not is_blank(thres[i:i+patch_size[1], j:j+patch_size[0]],
                            blank_threshold=blank_threshold):
                patches.append(gray_img[i:i+patch_size[1], j:j+patch_size[0]])
    return patches

def is_blank(patch, blank_threshold=1.0):
    '''
    Check path is blank (no characters)
    '''
    compare_matrix = patch == 255
    if blank_threshold >= 1:
        return np.all(compare_matrix)
    else:
        blank_ratio = np.count_nonzero(compare_matrix)/compare_matrix.size
        return blank_ratio >= blank_threshold or blank_ratio <= 1 - blank_threshold

def image_normalize(img):
    '''
    Normalize image by get mean and divide standard derivation
    '''
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1.0/np.sqrt(img.size))
    result = np.multiply(np.subtract(img, mean), 1/std_adj)
    if len(result.shape) == 2:
        result = np.expand_dims(result, -1)
    return result

def generate_patches_dataset(img_paths, eval_paths, output_directory):
    '''
    Generate patches from image paths to specific directory
    '''
    if len(img_paths) != len(eval_paths):
        raise ValueError('images and evaluates not the same')
    img_directory_path = os.path.join(output_directory, 'img')
    os.makedirs(img_directory_path)
    labels_file = open(os.path.join(output_directory, 'labels.txt'), 'w')
    index = 0
    for i, img_path in enumerate(img_paths):
        print('{}/{}'.format(i, len(img_paths)))
        img = cv2.imread(img_path)
        score = get_score_from_eval(eval_paths[i])
        patches = generate_patches(img, img_type='BGR')
        for patch in patches:
            filename = 'img_{}.jpg'.format(index)
            cv2.imwrite(
                os.path.join(img_directory_path, filename),
                patch)
            labels_file.write('{}\t{}\n'.format(filename, score))
            index += 1
        print('Generate {} patches'.format(len(patches)))
    labels_file.close()

def visualize_data(imgs, labels, rows, cols):
    '''
    Visualize document images dataset with labels
    '''
    for i in range(min(rows*cols), len(imgs)):
        plt.subplot(rows, cols, i)
        plt.imshow(imgs[i])
        plt.xticks([])
        plt.yticks([])
        plt.title('Score: {}'.format(labels[i]))
