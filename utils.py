import os
import random
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, corners):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(corners)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def is_rectangle(verticles):
    if len(verticles) != 4:
        return False
    for i in range(-1, 3):
        v1 = verticles[i - 1] - verticles[i]
        v2 = verticles[i + 1] - verticles[i]
        d_cosin = dist.cosine(v1, v2)
        if d_cosin < 0.75:
            return False
    return True

def get_document_corners(img, img_type='BGR'):
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
    img = cv2.copyMakeBorder(img, border_img, border_img, border_img, border_img, cv2.BORDER_CONSTANT)
    img = cv2.medianBlur(img, 11)
    edges = cv2.Canny(img, 10, 30)
    kernel = np.ones((3, 3),np.uint8)
    edges = cv2.dilate(edges, kernel, iterations = 1) # increase connect edges
    im, cnts, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    simplified_cnts = []
    for cnt in cnts:
        hull = cv2.convexHull(cnt)
        simplified_cnts.append(cv2.approxPolyDP(hull, 0.05*cv2.arcLength(hull, True), True))
    simplified_cnts = sorted(simplified_cnts, key=cv2.contourArea, reverse=True)
    area_th = cv2.contourArea(simplified_cnts[0])/5
    corners = None
    for i in range(1,min(5, len(simplified_cnts))):
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

def get_score_from_eval(filename):
    with open(filename, 'rb') as infile:
        lines = [line for line in infile][:4]
    return float(lines[3].split()[0][:-1]) / 100

def get_datasets(first_part_path, second_part_path):
    first_sets = os.listdir(first_part_path)
    image_paths = []
    eval_paths = [] 

    for doc in first_sets:
        files = os.listdir(os.path.join(first_part_path, doc))
        for item in files:
            split_item = os.path.splitext(item)
            if split_item[1].lower() == '.jpg':
                image_paths.append(os.path.join(first_part_path, doc, split_item[0] + split_item[1]))
                eval_paths.append(os.path.join(first_part_path, doc, 'eval_' + split_item[0] + '.txt'))
    second_part_path = os.path.join(second_part_path, 'FineReader')
    second_sets = os.listdir(second_part_path)
    for doc in second_sets:
        if doc[:3] == 'set':
            files = os.listdir(os.path.join(second_part_path, doc))
            for item in files:
                split_item = os.path.splitext(item)
                if split_item[1].lower() == '.jpg':
                    image_paths.append(os.path.join(second_part_path, doc, split_item[0] + split_item[1]))
                    eval_paths.append(os.path.join(second_part_path, doc, 'eval_' + split_item[0] + '.txt'))

    return image_paths, eval_paths

def generate_patches(img, patch_size=(48, 48), img_type='RGB', blank_threshold=1.0,
                     row_step=None, col_step=None):
    corners = get_document_corners(img, img_type)
    img = four_point_transform(img, corners)
    if img_type == 'BGR':
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img_type == 'RGB':
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img_type == 'GRAY':
        gray_img = img
    th = cv2.adaptiveThreshold(cv2.medianBlur(gray_img, 31), 255, \
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31,2)
    patches = []
    rows, cols = gray_img.shape
    if row_step is None:
        row_step = int(patch_size[1]/3)
    if col_step is None:
        col_step = int(patch_size[0]/3)
    for i in range(0, rows-patch_size[1], row_step):
        for j in range(0, cols-patch_size[0], col_step):
            if not is_blank(th[i:i+patch_size[1], j:j+patch_size[0]], blank_threshold=blank_threshold):
                patches.append(gray_img[i:i+patch_size[1], j:j+patch_size[0]])
    return patches

def is_blank(patch, blank_threshold=1.0):
    compare_matrix = patch == 255
    if blank_threshold >= 1:
        return np.all(compare_matrix)
    else:
        blank_ratio = np.count_nonzero(compare_matrix)/compare_matrix.size
        return blank_ratio >= blank_threshold or blank_ratio <= 1 - blank_threshold

def image_normalize(x):
    x = x.astype(np.float32)
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    if len(y.shape) == 2:
        y = np.expand_dims(y, -1)
    return y

def generate_patches_dataset(img_paths, eval_paths, dist):
    if len(img_paths) != len(eval_paths):
        raise ValueError('images and evaluates not the same')
    img_directory_path = os.path.join(dist, 'img')
    os.makedirs(img_directory_path)
    labels_file = open(os.path.join(dist, 'labels.txt'), 'w')
    index = 0
    for i in range(len(img_paths)):
        print('{}/{}'.format(i, len(img_paths)))
        img = cv2.imread(img_paths[i])
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
    for i in range(min(rows*cols), len(imgs)):
        plt.subplot(rows, cols, i)
        plt.imshow(imgs[i])
        plt.xticks([])
        plt.yticks([])
        plt.set_title('Score: {}'.format(labels[i]))

