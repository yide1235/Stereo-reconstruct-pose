
import cv2
import numpy as np


def StereoSGBM_create(minDisparity=None, numDisparities=None, blockSize=None, P1=None, P2=None,
                      disp12MaxDiff=None, preFilterCap=None, uniquenessRatio=None,
                      speckleWindowSize=None, speckleRange=None, mode=None):
    """

    :param mode:
    :return:
    """

def reprojectImageTo3D(disparity, Q, _3dImage=None, handleMissingValues=None, ddepth=None):

    return cv2.reprojectImageTo3D(disparity, Q, _3dImage, handleMissingValues, ddepth)


def get_depth(disparity, Q, scale=1.0, method=True):

    if method:
        points_3d = cv2.reprojectImageTo3D(disparity, Q) 
        x, y, depth = cv2.split(points_3d)
    else:
        # baseline = abs(camera_config["T"][0])
        baseline = 1 / Q[3, 2]  
        fx = abs(Q[2, 3])
        depth = (fx * baseline) / disparity
    depth = depth * scale
    # depth = np.asarray(depth, dtype=np.uint16)
    depth = np.asarray(depth, dtype=np.float32)
    return depth


class WLSFilter():
    def __init__(self, left_matcher, lmbda=80000, sigma=1.3):

        self.filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        self.filter.setLambda(lmbda)
        self.filter.setSigmaColor(sigma)

    def disparity_filter(self, dispL, imgL, dispR):
        filter_displ = self.filter.filter(dispL, imgL, None, dispR)
        return filter_displ


def get_filter_disparity(imgL, imgR, use_wls=True, sgbm="param2"):

    channels = 1 if imgL.ndim == 2 else 3
    blockSize = 3
    if sgbm == "param1":
        paramL = {"minDisparity": 0,
                  "numDisparities": 5 * 16,
                  "blockSize": blockSize,
                  "P1": 8 * 3 * blockSize,
                  "P2": 32 * 3 * blockSize,
                  "disp12MaxDiff": 12,
                  "uniquenessRatio": 10,
                  "speckleWindowSize": 50,
                  "speckleRange": 32,
                  "preFilterCap": 63,
                  "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY
                  }
    elif sgbm == "param2":
        paramL = {"minDisparity": 0,
                  "numDisparities": 5 * 16,
                  "blockSize": blockSize,
                  "P1": 8 * 3 * blockSize,
                  "P2": 32 * 3 * blockSize,
                  "disp12MaxDiff": 12,
                  "uniquenessRatio": 10,
                  "speckleWindowSize": 50,
                  "speckleRange": 32,
                  "preFilterCap": 63,
                  "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY
                  }
    else:
        paramL = {'minDisparity': 0,
                  'numDisparities': 128,
                  'blockSize': blockSize,
                  'P1': 8 * channels * blockSize ** 2,
                  'P2': 32 * channels * blockSize ** 2,
                  'disp12MaxDiff': 1,
                  'preFilterCap': 63,
                  'uniquenessRatio': 15,
                  'speckleWindowSize': 100,
                  'speckleRange': 1,
                  'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
                  }

    matcherL = cv2.StereoSGBM_create(**paramL)

    dispL = matcherL.compute(imgL, imgR)
    dispL = np.int16(dispL)

    if use_wls:

        matcherR = cv2.ximgproc.createRightMatcher(matcherL)
        dispR = matcherR.compute(imgR, imgL)
        dispR = np.int16(dispR)
        lmbda = 80000
        sigma = 1.3
        filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcherL)
        filter.setLambda(lmbda)
        filter.setSigmaColor(sigma)
        dispL = filter.filter(dispL, imgL, None, dispR)
        dispL = np.int16(dispL)

    dispL[dispL < 0] = 0
    dispL = dispL.astype(np.float32) / 16.
    return dispL


def get_simple_disparity(imgL, imgR):


    blockSize = 8
    channels = 3
    matcherL = cv2.StereoSGBM_create(minDisparity=1,
                                     numDisparities=64,
                                     blockSize=blockSize,
                                     P1=8 * channels * blockSize,
                                     P2=32 * channels * blockSize,
                                     disp12MaxDiff=-1,
                                     preFilterCap=1,
                                     uniquenessRatio=10,
                                     speckleWindowSize=100,
                                     speckleRange=100,
                                     mode=cv2.STEREO_SGBM_MODE_HH)

    dispL = matcherL.compute(imgL, imgR)
    dispL = np.int16(dispL)

    dispL[dispL < 0] = 0
    dispL = np.divide(dispL.astype(np.float32), 16.)
    return dispL


def get_visual_depth(depth, clip_max=6000):

    depth = np.clip(depth, 0, clip_max)

    depth = cv2.normalize(src=depth, dst=depth, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    depth = np.uint8(depth)
    depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth_colormap


def get_visual_disparity(disp, clip_max=6000):

    disp = np.clip(disp, 0, clip_max)
    disp = np.uint8(disp)
    return disp
