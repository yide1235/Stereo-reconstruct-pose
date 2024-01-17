
import cv2
import numpy as np


def getRectifyTransform(height, width, config):


    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T


    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


def rectify_image(imgL, imgR, map1x, map1y, map2x, map2y):

    rectifyed_img1 = cv2.remap(imgL, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(imgR, map2x, map2y, cv2.INTER_AREA)
    return rectifyed_img1, rectifyed_img2


def get_rectify_image(imgL, imgR):

    rectifiedL = cv2.remap(imgL, camera_config["left_map_x"], camera_config["left_map_y"],
                           interpolation=cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
    rectifiedR = cv2.remap(imgR, camera_config["right_map_x"], camera_config["right_map_y"],
                           interpolation=cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
    return rectifiedL, rectifiedR


def draw_line_rectify_image(imgL, imgR, interval=50, color=(0, 255, 0), show=False):

    height = max(imgL.shape[0], imgR.shape[0])
    width = imgL.shape[1] + imgR.shape[1]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[0:imgL.shape[0], 0:imgL.shape[1]] = imgL
    img[0:imgR.shape[0], imgL.shape[1]:] = imgR

    for k in range(height // interval):
        cv2.line(img, (0, interval * (k + 1)), (2 * width, interval * (k + 1)), color, 2, lineType=cv2.LINE_AA)
    if show:
        cv2.imshow('rectify_image', img)
        cv2.waitKey(1)
    return img
