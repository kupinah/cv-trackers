import math

import cv2
import numpy as np


def draw_region(self, img, region, color, line_width):
    if len(region) == 4:
        # rectangle
        tl = (int(round(region[0])), int(round(region[1])))
        br = (int(round(region[0] + region[2] - 1)), int(round(region[1] + region[3])))
        cv2.rectangle(img, tl, br, color, line_width)
    elif len(region) == 8:
        # polygon
        pts = np.round(np.array(region).reshape((-1, 1, 2))).astype(np.int32)
        cv2.polylines(img, [pts], True, color, thickness=line_width, lineType=cv2.LINE_AA)
    else:
        print("Error: Unknown region format.")
        exit(-1)


def custom_get_patch(x, y, h, w, img):  # X, Y are top left corner of the patch!!!
    x = max(int(x), 0)
    y = max(int(y), 0)

    x = min(x, img.shape[1] - w)
    y = min(y, img.shape[0] - h)

    if x + w >= img.shape[1]:
        x = img.shape[1] - w - 1
    if y + h >= img.shape[0]:
        y = img.shape[0] - h - 1

    sr = img[y : y + h, x : x + w]

    return x, y, sr


def create_cosine_window(target_size):
    # target size is in the format: (width, height)
    # output is a matrix of dimensions: (width, height)
    return cv2.createHanningWindow((target_size[0], target_size[1]), cv2.CV_32F)


def create_gauss_peak(target_size, sigma):
    # target size is in the format: (width, height)
    # sigma: parameter (float) of the Gaussian function
    # note that sigma should be small so that the function is in a shape of a peak
    # values that make sens are approximately from the interval: ~(0.5, 5)
    # output is a matrix of dimensions: (width, height)
    w2 = math.floor(target_size[0] / 2)
    h2 = math.floor(target_size[1] / 2)
    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))

    if X.shape != (target_size[1], target_size[0]):
        X = X[0 : target_size[1], 0 : target_size[0]]
        Y = Y[0 : target_size[1], 0 : target_size[0]]

    G = np.exp(-(X**2) / (2 * sigma**2) - Y**2 / (2 * sigma**2))
    G = np.roll(G, (-h2, -w2), (0, 1))
    return G
