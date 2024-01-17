
import numpy as np
import cv2
from core.utils.file_storage import load_stereo_coefficients


def get_rectify_transform(K1, D1, K2, D2, R, T, image_size):

    # R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, alpha=0)
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T,
                                                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)

    left_map_x, left_map_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    return left_map_x, left_map_y, right_map_x, right_map_y, R1, R2, P1, P2, Q


def get_stereo_coefficients(stereo_file, rectify=True):

    # Get cams params
    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q, size = load_stereo_coefficients(stereo_file)
    config = {}
    config["size"] = size  
    config["K1"] = K1  
    config["D1"] = D1 
    config["K2"] = K2  
    config["D2"] = D2  
    config["R"] = R  
    config["T"] = T  
    config["E"] = E 
    config["F"] = F  
    config["R1"] = R1
    config["R2"] = R2
    config["P1"] = P1
    config["P2"] = P2
    config["Q"] = Q
    if rectify:
   
        left_map_x, left_map_y, right_map_x, right_map_y, R1, R2, P1, P2, Q = get_rectify_transform(K1, D1, K2, D2,
                                                                                                    R, T, size)
        config["R1"] = R1
        config["R2"] = R2
        config["P1"] = P1
        config["P2"] = P2
        config["Q"] = Q
        config["left_map_x"] = left_map_x
        config["left_map_y"] = left_map_y
        config["right_map_x"] = right_map_x
        config["right_map_y"] = right_map_y
    focal_length = Q[2, 3]  

    baseline = 1 / Q[3, 2]  
    print("Q=\n{}\nfocal_length={}".format(Q, focal_length))
    print("T=\n{}\nbaseline    ={}mm".format(T, baseline))
    return config



class stereoCamera(object):
    def __init__(self, width=640, height=480):
  

        self.cam_matrix_left = np.asarray([[4.1929128272967574e+02, 0., 3.2356123553538390e+02],
                                           [0., 4.1931862286777556e+02, 2.1942548262685406e+02],
                                           [0., 0., 1.]])

        self.cam_matrix_right = np.asarray([[4.1680693687859372e+02, 0., 3.2769747052057716e+02],
                                            [0., 4.1688284886037280e+02, 2.3285709632482832e+02],
                                            [0., 0., 1.]])


        self.distortion_l = np.asarray([[-2.9558582315073436e-02,
                                         1.5948145293240729e-01,
                                         -7.1046620767870137e-04,
                                         -6.5787270354389317e-04,
                                         -2.7169829618300961e-01]])

        self.distortion_r = np.asarray([[-2.3391571805264716e-02,
                                         1.3648437316647929e-01,
                                         6.7233698457319337e-05,
                                         5.8610808515832777e-04,
                                         -2.3463198941301094e-01]])


        self.R = np.asarray([[9.9995518261153071e-01, 4.2888473189297411e-04, -9.4577389595457383e-03],
                             [-4.4122271031099070e-04, 9.9999905442083736e-01, -1.3024899043586808e-03],
                             [9.4571713984714298e-03, 1.3066044993798060e-03, 9.9995442630843034e-01]])

  
        self.T = np.asarray([[-2.2987774547369614e-02], [3.0563972870288424e-05], [8.9781163185012418e-05]])

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.cam_matrix_left,
                                                          self.distortion_l,
                                                          self.cam_matrix_right,
                                                          self.distortion_r,
                                                          (width, height),
                                                          self.R,
                                                          self.T,
                                                          alpha=0)


        self.focal_length = Q[2, 3]  


        self.baseline = self.T[0]  


if __name__ == "__main__":
    stereo_file = "../config/main_camera/stereo_cam.yml"
    config = get_stereo_coefficients(stereo_file, width=640, height=480)
