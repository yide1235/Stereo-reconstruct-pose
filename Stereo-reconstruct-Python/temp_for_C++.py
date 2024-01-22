import sys
import os
import cv2
import argparse
import numpy as np
from core.utils import image_utils, file_utils
from core import camera_params, stereo_matcher
sys.path.append('./YOLOV8')
sys.path.append('./THUNDER/')
sys.path.append('./LIGHT/')
from yolov8 import run_yolo
from thunder import run_thunder
from light import run_light
import random
import statistics
import re
import math
import os
import shutil

from math import *
import statistics

def phi(z):
    #'Cumulative distribution function for the standard normal distribution'
    ##pass the z value this will return cumulative 1 tailed probability to Z
    return (1.0 + erf(z / sqrt(2.0))) / 2.0

def point_match(dist1_mu, dist1_sig, val):
    ##takes a two tailed match of how close the value is to the mean.
    ## eg. 1 - P value
    z = (val - dist1_mu) / dist1_sig
    z = abs(z)  ## flip it around so always more than 0.5
    cum_prob = phi(z)
    return 1 - cum_prob
    
    


def dist_intersect(dist1_mu, dist1_sig, dist2_mu, dist2_sig):
    #returns the intersection probability of two distributions
    step_sig = max(dist1_sig, dist2_sig)
    start_mu = min(dist1_mu, dist2_mu)
    end_mu = max(dist1_mu, dist2_mu)
    step = 6 * step_sig / 10

    #print("mu diff in sigma is ", abs(dist1_mu  - dist2_mu)/min(dist1_sig, dist2_sig))
    #print(dist1_mu , dist2_mu,  dist1_sig, dist2_sig)

    #if abs(dist1_mu  - dist2_mu) > 6 * min(dist1_sig, dist2_sig):
    #    return 0.01  ##return a small positive value if the two distributions are too far apart

    startx = start_mu - 6 * step_sig
    endx = end_mu + 6 * step_sig
    currx = startx
    int_prob = 0  ##intersection probability
    while(currx < endx):
        refz1 = (currx - dist1_mu) / dist1_sig
        refz2 = ((currx+step) - dist1_mu) / dist1_sig
        p1 = phi(refz1)
        p2 = phi(refz2)
        prob1 = abs(p2-p1)

        refz1 = (currx - dist2_mu) / dist2_sig
        refz2 = ((currx+step) - dist2_mu) / dist2_sig
        p1 = phi(refz1)
        p2 = phi(refz2)
        prob2 = abs(p2-p1)

        int_prob += min(prob1,prob2)
        currx += step

    return int_prob

def est_sig(rng, prob, map_probs):
    ##estimates the sigma by determining the standard normal two
    prob = round(prob,2)
    if prob in map_probs.keys():
        #print(" range is ", rng, " prob is ", prob, " sigma is ", map_probs[prob] * rng)
        return 0.5 * rng / map_probs[prob]  ##this is the estimate of sigma
    else:
        if prob <= 0 or prob > 1:
            return None
    

def gen_dict():
    map_probs = {}
    for x in range(1,350):
        prob = round(phi(x/100)-phi(-x/100),2)
        #print("x:", x/100, ", prob:", round(phi(x/100)-phi(-x/100),2) , ",")
        if prob not in map_probs.keys():
            map_probs[prob] = x/100

    return map_probs

def intersect(means1, means2, range1, range2, prob, map_probs):
    ##if collecting middle 3 out of 7 values, prob is 3/7
    ##means1 is the 12 means of the 3 points above, range1 is the +/- range of the 3 points from above
    ##means2 is the 12 means of the 3 points above, range1 is the +/- range of the 3 points from above
    mult = 1
    probs = []
    cnt = 0
    nan_cnt = 0
    tot = 0

    mean_count1 = 0
    mean_count2 = 0
    for x in range(0,len(means1)):
        sig1 = est_sig(range1[x], prob, map_probs)
        sig2 = est_sig(range2[x], prob, map_probs)
        if sig1 == 0 or sig2 ==0:
            probs.append("NAN")
            nan_cnt += 1
            continue;
        
        int_prob = dist_intersect(means1[x], sig1, means2[x], sig2)
        if int_prob == 0:
            probs.append("NAN")
            nan_cnt += 1
            continue;
        #print(sig1, sig2, "int_prob", int_prob)
        mult *= int_prob
        tot += int_prob
        cnt += 1
        probs.append(int_prob)
        mean_count1 += means1[x]
        mean_count2 += means2[x]

    avg_prob = tot / cnt
    if nan_cnt > 0:
        mult *= 10**(-1*nan_cnt)

    ##mult = ((mean_count2 - mean_count1)**2/(mean_count1+mean_count2)) * mult 

    # print("mult is ", mult, probs)
    return mult, probs


def intersect_or(means1, means2, range1, range2, prob, map_probs):
    ##if collecting middle 3 out of 7 values, prob is 3/7
    ##means1 is the 12 means of the 3 points above, range1 is the +/- range of the 3 points from above
    ##means2 is the 12 means of the 3 points above, range1 is the +/- range of the 3 points from above
    mult = 1
    probs = []
    cnt = 0
    nan_cnt = 0
    tot = 0

    mean_count1 = 0
    mean_count2 = 0
    for x in range(0,len(means1)):
        sig1 = est_sig(range1[x], prob, map_probs)
        sig2 = est_sig(range2[x], prob, map_probs)
        if sig1 == 0 or sig2 ==0:
            probs.append("NAN")
            nan_cnt += 1
            continue;
        
        int_prob = dist_intersect(means1[x], sig1, means2[x], sig2)
        if int_prob == 0:
            probs.append("NAN")
            nan_cnt += 1
            continue;
        #print(sig1, sig2, "int_prob", int_prob)
        mult += int_prob
        tot += int_prob
        cnt += 1
        probs.append(int_prob)
        mean_count1 += means1[x]
        mean_count2 += means2[x]

    mult = mult / cnt

    return mult, probs


#above intersection function




class StereoDepth(object):

    #start of triangulation
    def __init__(self, stereo_file, width=1920, height=1080, filter=True, use_open3d=False, use_pcl=False):

        self.count = 0
        self.filter = filter
        self.camera_config = camera_params.get_stereo_coefficients(stereo_file)
        self.use_pcl = use_pcl
        self.use_open3d = use_open3d
    
        if self.use_pcl:
 
            from core.utils_pcl import pcl_tools
            self.pcl_viewer = pcl_tools.PCLCloudViewer()

        assert (width, height) == self.camera_config["size"], Exception("Error:{}".format(self.camera_config["size"]))




    def get_3dpoints(self, disparity, Q, scale=1.0):
        '''
        :param Q:Q=[[1, 0, 0, -cx]
                           [0, 1, 0, -cy]
                           [0, 0, 0,  f]
                           [1, 0, -1/Tx, (cx-cx`)/Tx]]
        '''

        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        # x, y, depth = cv2.split(points_3d)
        # baseline = abs(camera_config["T"][0])
        # baseline = 1 / Q[3, 2] 
        # fx = abs(Q[2, 3])
        # depth = (fx * baseline) / disparity
        points_3d = points_3d * scale
        points_3d = np.asarray(points_3d, dtype=np.float32)
        return points_3d


    def get_disparity(self, imgL, imgR, use_wls=True):

        dispL = stereo_matcher.get_filter_disparity(imgL, imgR, use_wls=use_wls)
        # dispL = disparity.get_simple_disparity(imgL, imgR)
        return dispL

    def get_rectify_image(self, imgL, imgR):


        # camera_params.get_rectify_transform(K1, D1, K2, D2, R, T, image_size)
        left_map_x, left_map_y = self.camera_config["left_map_x"], self.camera_config["left_map_y"]
        right_map_x, right_map_y = self.camera_config["right_map_x"], self.camera_config["right_map_y"]
        rectifiedL = cv2.remap(imgL, left_map_x, left_map_y, cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
        rectifiedR = cv2.remap(imgR, right_map_x, right_map_y, cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
        return rectifiedL, rectifiedR

    #end of triangulation



    def run_pose_onetime(self, rectifiedL=None,bbox=None,base_filename=None,box_num=None):
        #not doing augmentation
        rectified_left_points_results=[]
        rectified_left_points = []
        conf_sums_local=[]
        

        # Extract and save the cropped image from the bounding box
        cropped_img = rectifiedL[max(0, int(bbox[1] - 0.05 * (bbox[3] - bbox[1]))):int(min(rectifiedL.shape[0], bbox[3] + 0.05 * (bbox[3] - bbox[1]))), max(0, int(bbox[0] - 0.05 * (bbox[2] - bbox[0]))):min(rectifiedL.shape[1], int(bbox[2] + 0.05 * (bbox[2] - bbox[0])))]
        
        cv2.imwrite('{}_box_l_aug_{}.jpg'.format(base_filename,box_num), cropped_img)

        p_l,conf_sum= run_light('{}_box_l_aug_{}.jpg'.format(base_filename,box_num))
        # p_l,conf_sum= run_thunder('{}_box_l_aug_{}.jpg'.format(base_filename,box_num))

        conf_sums_local.append(conf_sum)


        for point in p_l[0]:

            if point[0] == point[1] == 0:
                points_l = [[0, 0]]
            else:

                x_adj=point[0] + max(0, int(bbox[0] - 0.05 * (bbox[2] - bbox[0])))
                y_adj=point[1] + max(0, int(bbox[1] - 0.05 * (bbox[3] - bbox[1])))


                points_l = [[int(x_adj), int(y_adj)]]

            rectified_left_points.append(points_l)


        rectified_left_points_result=rectified_left_points

        rectified_left_points_results.append(rectified_left_points_result)

        return rectified_left_points_results, conf_sums_local


        

    def run_pose_aug(self, rectifiedL=None,bbox=None, base_filename=None,box_num=None):

        rectified_left_points_results=[]

        for g in range(num_aug):

            # Step 1: Load the image

            height, width = rectifiedL.shape[:2]

            imgl_temp = rectifiedL.copy()
            # Step 2: Apply random augmentation
            # Randomly choose an augmentation: noise, stretching, or cropping
            # augmentation = random.choice(['noise', 'stretch','crop'])

            # augmentation = random.choice(['noise', 'stretch'])

            augmentation = random.choice(['noise'])

            # augmentation = random.choice(['stretch'])

            # Initialize transformation matrix
            M = np.eye(3)

            if augmentation == 'noise':
                # Check the number of channels in the image
                if len(imgl_temp.shape) == 3:
                    # If the image is color (BGR), create noise with 3 channels
                    noise = np.random.randint(0, 50, imgl_temp.shape, dtype='uint8')
                else:
                    # If the image is grayscale, create single channel noise
                    noise = np.random.randint(0, 50, (height, width), dtype='uint8')

                imgl_temp = cv2.add(imgl_temp, noise)

            elif augmentation == 'stretch':
                # Randomly choose stretch factors
                fx, fy = random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)
                imgl_temp = cv2.resize(imgl_temp, None, fx=fx, fy=fy)
                M = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])

            # elif augmentation == 'crop':
            #     # Randomly choose crop size
            #     x, y = random.randint(0, width // 4), random.randint(0, height // 4)
            #     w, h = width - 2*x, height - 2*y
            #     imgl_temp = imgl_temp[y:y+h, x:x+w]
            #     M = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])


            # Step 3: Detect keypoints using the augmented image
            rectified_left_points = []
            conf_sums_local=[]
            conf_sum=0

            cropped_img = imgl_temp[max(0, int(bbox[1] - 0.05 * (bbox[3] - bbox[1]))):int(min(imgl_temp.shape[0], bbox[3] + 0.05 * (bbox[3] - bbox[1]))), max(0, int(bbox[0] - 0.05 * (bbox[2] - bbox[0]))):min(imgl_temp.shape[1], int(bbox[2] + 0.05 * (bbox[2] - bbox[0])))]
            cv2.imwrite('box_l_aug_{}.jpg'.format(box_num), cropped_img)

            p_l,conf_sum = run_thunder('box_l_aug_{}.jpg'.format(box_num))
            # p_l,conf_sum = run_light('box_l_aug_{}.jpg'.format(box_num))


            conf_sums_local.append(conf_sum)

            # print(p_l)

            for point in p_l[0]:

                if point[0] == point[1] == 0:
                    points_l = [[0, 0]]
                else:
                    # Adjust keypoints based on augmentation
                    x_adj, y_adj = point[0] + max(0, int(bbox[0] - 0.05 * (bbox[2] - bbox[0]))), point[1] + max(0, int(bbox[1] - 0.05 * (bbox[3] - bbox[1])))

                    # Transform points back according to the inverse of M
                    if augmentation != 'noise':
                        inv_M = np.linalg.inv(M)
                        x_adj, y_adj, _ = np.dot(inv_M, [x_adj, y_adj, 1])

                    points_l = [[int(x_adj), int(y_adj)]]

                rectified_left_points.append(points_l)

            rectified_left_points_results.append(rectified_left_points)
        
        return rectified_left_points_results, conf_sums_local


        #end of 50 loops








    def task(self, frameL, frameR, waitKey=5, show_depth=False, use_augmentation=None, left_file_path=None, num_aug=None, output_folder=None, pick_biggest_box=True,run_onetime=None,effective_range=None,max_min=None):


        rectifiedL, rectifiedR = self.get_rectify_image(imgL=frameL, imgR=frameR)

        cv2.imwrite('./rect_l.jpg',rectifiedL)
        cv2.imwrite('./rect_r.jpg',rectifiedR)

        base_filename = os.path.basename(left_file_path).split('.')[0]
        # output_path = f'./same_3persons_results/{base_filename}_processed.jpg'
        output_path = f'./{output_folder}/{base_filename}_processed.jpg'


        if not show_depth:
        # #code for single point
            
            

            #calculate the disparity for onetime
            grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)

            dispL = self.get_disparity(grayL, grayR, self.filter)
            points_3d = self.get_3dpoints(disparity=dispL, Q=self.camera_config["Q"])
            # self.show_2dimage(frameL, frameR, points_3d, dispL, waitKey=waitKey)
            # print(points_3d)

            x, y, depth = cv2.split(points_3d) 
            xyz_coord = points_3d  # depth = points_3d[:, :, 2]
            
            depth_colormap = stereo_matcher.get_visual_depth(depth)
            dispL_colormap = stereo_matcher.get_visual_disparity(dispL)




            rectified_left_points_results=[]
            conf_sums=[]
            front=False
            fronts=[]
            in_effective_range=False
            # center_point=None

            bboxes_l=run_yolo(['rect_l.jpg'])

            # print(bboxes_l)
            box_num=0



            #this variable is for do the median on 2d points
            get_average=False
            #if this set to True then the mag will be one


            #this variable is for do the median on 3d mag
            get_average_mag=True


            # print(bboxes_l)
            if pick_biggest_box:
                bbox_size=0
                for i in bboxes_l:
                    temp=(i[3]-i[1])*(i[2]-i[0])
                    if temp>bbox_size:
                        bbox_size=temp
                        bboxes_l=[np.array(i)]
                        # center_point=[(i[2]-i[0])/2, (i[3]-i[1])/2]

            
            # print(bboxes_l)
            bbox_index=0
            rectifiedL_copy = rectifiedL.copy()


            means_onebox=None
            ranges_onebox=None
            temp_final_onebox=None


            variance_threshold=2.5

            #assume one bbox for now
            # for bbox in bboxes_l:
            if len(bboxes_l)>0:
                bbox=bboxes_l[0]

                conf_sums_local=[]
                # rectified_left_points_results,conf_sums_local = self.run_pose_onetime(rectifiedL=rectifiedL,bbox=bbox,base_filename=base_filename,box_num=box_num)
                rectified_left_points_results,conf_sums_local = self.run_pose_aug(rectifiedL=rectifiedL,bbox=bbox,base_filename=base_filename,box_num=box_num)


                # print(rectified_left_points_results)

                #if works continue, if not return None


                #check if it front or back
                #if right shoulder 6 is smaller than the left shoulder 5, then this is front, otherwise it is back
                for rectified_left_points_result in rectified_left_points_results:
                    if rectified_left_points_result[6][0][0]<rectified_left_points_result[5][0][0]:
                        front=True
                    else:
                        front=False

                fronts.append(front)   
                conf_sums=sum(conf_sums_local)/num_aug
                box_num+=1


                depth_3d=[]

                print(rectified_left_points_results)

                for rectified_left_points_result in rectified_left_points_results:
                    for k in range(17):
                        tempx=rectified_left_points_result[k][0][0]
                        tempy=rectified_left_points_result[k][0][1]

                        points3d=xyz_coord[round(tempx)][round(tempy)]
                        depth_3d.append(points3d[2])

                # print(depth_3d)
                result_depth=(depth_3d)
                # print(result_depth)
                # points3d=xyz_coord[round(tempx)][round(tempy)]
                # # points3d=xyz_coord[round(rectified_left_points_results[0][5][0][0])][round(rectified_left_points_results[0][5][0][1])]
                # # print(tempx, tempy)
                # # print(points3d)
                # print(points3d[0], points3d[1], points3d[2])

                print(result_depth)
                print('111111111111111111', result_depth<effective_range)
                print('111111111111111111', conf_sums_local[0]>max_min)
                if result_depth<effective_range and conf_sums_local[0]>max_min:

                    
                
                    if get_average:
                        # print('----',rectified_left_points_results)
                        #here assume just one person
                        rectified_left_points_results=[calculate_median(rectified_left_points_results)]
                        # print('----', rectified_left_points_results)

                    # rectifiedL_copy = rectifiedL.copy()


                    

                    
                    for rectified_left_points_result in rectified_left_points_results:
                        for i in rectified_left_points_results:
                            for pr in i:

                                center = [round(pr[0][0]), round(pr[0][1])]
                                cv2.circle(rectifiedL_copy, center, 2, (0, 255, 0), -1)
                        
                        
                        x=rectified_left_points_result[6][0][0]
                        y=rectified_left_points_result[6][0][1]

                        text_to_draw=str(bbox_index)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_position = (x, y)  # Change it as per your requirement

                        # Font scale and color
                        font_scale = 1
                        font_color = (255, 0, 0) # Blue color in BGR

                        # Line type
                        line_type = 2
                        cv2.putText(rectifiedL_copy, text_to_draw, text_position, font, font_scale, font_color, line_type)

                    # cv2.imwrite(output_path, rectifiedL_copy)

                    vecs=[]

                    # for rectified_left_points_results_each_aug in rectified_left_points_results:




                    
                    for rectified_left_points_result in rectified_left_points_results:
                        points3d=[]

                        for i in range(len(rectified_left_points_result)):

                            x=rectified_left_points_result[i][0][0]
                            y=rectified_left_points_result[i][0][1]

                            point3d=xyz_coord[round(y)][round(x)]

                            points3d.append(point3d)
                        #end for 17points of each person

                        print(points3d)

                        vec_inds = [[6, 5], [6, 8], [8, 10], [5, 7], [7, 9], [12, 14], [14, 16], [11, 13], [13, 15],[6,12],[5,11],[12,11]]
            
                        vec = []
                
                        for pair in vec_inds:
                            if np.all(points3d[pair[0]] == 0) or np.all(points3d[pair[1]] == 0):
                                #If either of the points is (0, 0, 0)
                                vec.append(0)
                            else:
                                # Calculate Euclidean distance and append

                                result = np.linalg.norm(points3d[pair[0]] - points3d[pair[1]]) / 10.0
                                vec.append(result)
                        
                    
                        vecs.append(vec)

            
                    vecs=vecs
                    vecs_final=[]

                    for i in vecs:
                        temp=[]
                        for j in i:
                            if math.isnan(j) or math.isinf(j):
                                temp.append(0)
                            else:
                                temp.append(j)
                        vecs_final.append(temp)
                    

                    # print(vecs_final)
                    means=[]
                    ranges=[]
                    temp_final=None
                    if get_average_mag:
                        # print(vecs_final)
                        temp=[]
                        for i in range(12):
                            temp2=[]
                            for j in vecs_final:
                                temp2.append(j[i])
                            temp.append(temp2)

                        # print(temp)
                        # temp_final=[]
                        # for i in temp:
                        #     temp3=[]
                        #     for j in i:
                                
                        #         if (not math.isnan(j)) or (not math.isinf) or (not 0.0):
                        #             temp3.append(j)
                        #     temp_final.append(temp3)
                        
                        # print('--------')
                        # print(temp)
                        temp_final= [sorted(list(filter(lambda x: x != 0, sublist))) for sublist in temp]
                        # print(temp_final)
                        std_vector=[]
                        for i in temp_final:
                            if len(i)>2:
                                std_vector.append(statistics.stdev(i))
                            else:
                                std_vector.append(-1)
                        # print(std_vector)
                        cout=0
                        for i in std_vector:
                            if i>0 and i<variance_threshold:
                                cout+=1

                        # print('how many valid digits it contains: ' , cout)

                        

                        #now delelte 0s for temp

                        if(cout>=9):
                            for i in temp_final:
                                if len(i)==0:
                                    mean=0
                                    range_temp=0
                                elif len(i)==1:
                                    mean=i[0]
                                    range_temp=0.5
                                elif len(i)==2:
                                    mean=(i[0]+i[1])/2
                                    range_temp=i[1]-i[0]
                                elif len(i)==3:
                                    mean=i[1]
                                    range_temp=i[2]-i[0]
                                elif len(i)==4:
                                    mean=(i[1]+i[2])/2
                                    range_temp=i[3]-i[0]
                                elif len(i)==5:
                                    mean=i[2]
                                    range_temp=i[3]-i[1]
                                else:
                                    low=round(len(i)*0.25)
                                    up=round(len(i)*0.75)
                                    # mean=round(i[round(len(i)/2)])
                                    mean=i[round(len(i)/2)]
                                    range_temp=i[up]-i[low]

                                #lower bound
                                if (range_temp is not 0) and (range_temp < 0.5):
                                    range_temp=0.5

                                #upper bound
                                if (range_temp > 4):
                                    range_temp=4                       

                                means.append(mean)
                                ranges.append(range_temp)
                                
                            # #if for that position of variance vector is>1.5 or <0, then that index should be dropped for mean and range
                            # print('---------------------')
                            # print(means)
                            # print(ranges)
                            means_final=[]
                            ranges_final=[]

                            for i in range(len(std_vector)):
                                if std_vector[i]>0 and std_vector[i]<variance_threshold:
                                    # means[i]==0
                                    # ranges[i]==0
                                    means_final.append(means[i])
                                    ranges_final.append(ranges[i])
                                else:
                                    means_final.append(0)
                                    ranges_final.append(0)

                            # print('new means and ranges is: ')
                            # print(means_final)
                            # print(ranges_final)



                            # assert 1== 0


                        else:
                            # print("No enough valid digits!!")
                            means=None
                            ranges=None
                            temp_final=None



                    means_onebox=means
                    ranges_onebox=ranges
                    temp_final_onebox=temp_final

                    # print('mean final for one bbox is: ', means_onebox)
                    # print('ranges final for one bbox is: ', ranges_onebox)
                    # print('parts final for on bbox is: ', temp_final_onebox)


                        # bbox_index=bbox_index+1

                    


                    cv2.imwrite(output_path, rectifiedL_copy)


                    #note that front and conf_sums might be wrong
                    return means_onebox, ranges_onebox, temp_final_onebox, front,conf_sums
                    # return vecs,front,conf_sums

                else:
                    # print("in uneffective range")
                    return None, None, None, None, None
            else:
                return None, None, None, None, None



        else:

            print('drawing the depths')
            # calibrate_tools.draw_line_rectify_image(rectifiedL, rectifiedR)
            # We need grayscale for disparity map.
            grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)
            # Get the disparity map
            dispL = self.get_disparity(grayL, grayR, self.filter)
            points_3d = self.get_3dpoints(disparity=dispL, Q=self.camera_config["Q"])
            
            # self.show_3dcloud_for_pcl(frameL, frameR, points_3d)

            self.show_2dimage(frameL, frameR, points_3d, dispL, waitKey=waitKey)

            return None
#ignore from here
    #ignore
    def show_3dcloud_for_pcl(self, frameL, frameR, points_3d):

        if self.use_pcl:
            self.pcl_viewer.add_3dpoints(points_3d/1000, frameL)
            self.pcl_viewer.show()

    #ignore
    def show_2dimage(self, frameL, frameR, points_3d, dispL, waitKey=0):
        """
        :param frameL:
        :param frameR:
        :param dispL:
        :param points_3d:
        :return:
        """
        x, y, depth = cv2.split(points_3d)  # depth = points_3d[:, :, 2]
        xyz_coord = points_3d  # depth = points_3d[:, :, 2]
        depth_colormap = stereo_matcher.get_visual_depth(depth)
        dispL_colormap = stereo_matcher.get_visual_disparity(dispL)
        image_utils.addMouseCallback("left", xyz_coord, info="world coords=(x,y,depth)={}mm",depth=depth_colormap,disparity=dispL_colormap)
        image_utils.addMouseCallback("right", xyz_coord, info="world coords=(x,y,depth)={}mm",depth=depth_colormap,disparity=dispL_colormap)
        image_utils.addMouseCallback("disparity-color", xyz_coord, info="world coords=(x,y,depth)={}mm",depth=depth_colormap,disparity=dispL_colormap)
        image_utils.addMouseCallback("depth-color", xyz_coord, info="world coords=(x,y,depth)={}mm",depth=depth_colormap,disparity=dispL_colormap)
        result = {"frameL": frameL, "frameR": frameR, "disparity": dispL_colormap, "depth": depth_colormap}
        
        # cv2.imshow('left', frameL)
        # cv2.imshow('right', frameR)
        cv2.imshow('disparity-color', dispL_colormap)
        cv2.imshow('depth-color', depth_colormap)
        
        key = cv2.waitKey(waitKey)
        self.save_images(result, self.count, key)
        if self.count <= 1:
            # cv2.moveWindow("left", 700, 0)
            # cv2.moveWindow("right", 1400, 0)
            cv2.moveWindow("disparity-color", 700, 700)
            cv2.moveWindow("depth-color", 1400, 700)
            cv2.waitKey(0)


    def save_images(self, result, count, key, save_dir="./data/temp"):
        """
        :param result:
        :param count:
        :param key:
        :param save_dir:
        :return:
        """
        if key == ord('q'):
            exit(0)
        elif key == ord('c') or key == ord('s'):
            file_utils.create_dir(save_dir)
            # print("save image:{:0=4d}".format(count))
            cv2.imwrite(os.path.join(save_dir, "left_{:0=4d}.png".format(count)), result["frameL"])
            cv2.imwrite(os.path.join(save_dir, "right_{:0=4d}.png".format(count)), result["frameR"])
            cv2.imwrite(os.path.join(save_dir, "disparity_{:0=4d}.png".format(count)), result["disparity"])
            cv2.imwrite(os.path.join(save_dir, "depth_{:0=4d}.png".format(count)), result["depth"])



def normalize_vector(float_vector):
    if not float_vector or max(float_vector) == 0:
        return float_vector
    else:
        max_value = max(i for i in float_vector if i != float('inf') and not math.isnan(i))
        return [(i / max_value if i != float('inf') and not math.isnan(i) else 0) for i in float_vector]



def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')



def get_parser():
    stereo_file = "configs/lenacv-camera/stereo_cam.yml"
    # stereo_file = "configs/lenacv-camera/stereo_matlab.yml"
    left_video = None
    right_video = None
    # left_video = "data/lenacv-video/left_video.avi"
    # right_video = "data/lenacv-video/right_video.avi"



    # left_file='./calib-new/44/left/frame00005.jpg'
    # right_file='./calib-new/44/right/frame00005.jpg'

    left_file=None
    right_file=None


    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--stereo_file', type=str, default=stereo_file, help='stereo calibration file')
    parser.add_argument('--left_video', default=left_video, help='left video file or camera ID')
    parser.add_argument('--right_video', default=right_video, help='right video file or camera ID')
    parser.add_argument('--left_file', type=str, default=left_file, help='left image file')
    parser.add_argument('--right_file', type=str, default=right_file, help='right image file')
    parser.add_argument('--filter', type=str2bool, nargs='?', default=True, help='use disparity filter')
    return parser

def median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)

    if n % 2 == 0:  # even number of elements
        return (sorted_lst[n//2 - 1] + sorted_lst[n//2]) / 2
    else:  # odd number of elements
        return sorted_lst[n//2]



def sort_numerically(file_name):
    """ Extracts the number from the filename and returns it for sorting. """
    numbers = re.findall(r'\d+', file_name)
    return int(numbers[0]) if numbers else 0



class Person:
    def __init__(self):
        # self.magnitude_vector = [0] * 24

        #person2
        # self.magnitude_vector=[32.42, 22.66, 21.1, 24.02, 20.53, 36.04, 0.01, 35.81, 0.01, 48.69, 45.36, 20.38, 29.94, 27.77, 0.01, 26.05, 21.63, 34.38, 0.01, 36.88, 0.01, 49.29, 48.85, 19.99]

        self.mature_vector = [0] * 24 
        self.placeholder = [[] for _ in range(24)] 
        self.macthed_count=0


    # def update(self,frame_vector, front, threshold):
    #     if front:
    #         for i in range(len(frame_vector)):
    #             self.placeholder[i].append(frame_vector[i])
    #     else:
    #         for i in range(len(frame_vector)):
    #             self.placeholder[i+12].append(frame_vector[i])
    #     temp_count=0
    #     # rounded_list = [0 if math.isinf(num) or math.isnan(num) else round(num) for num in frame_vector]
    #     rounded_list = [0 if math.isinf(num) or math.isnan(num) else num for num in frame_vector]
    #     print('this frame vector is: ',rounded_list)
    #     if front:


    #         for i in range(len(rounded_list)):
    #             if abs(rounded_list[i]-self.magnitude_vector[i])<1.5 and rounded_list[i]!=0:
    #                 temp_count+=1
    #     else:
    #         for i in range(len(rounded_list)):
    #             if abs(rounded_list[i]-self.magnitude_vector[i+12])<1.5 and rounded_list[i]!=0:
    #                 temp_count+=1
    #     print('match count for this frame is: ',temp_count)
    #     if temp_count>6:
    #         self.macthed_count+=1

    #     self.detect_mature(threshold)
    

    # def detect_mature(self,threshold):
    #     for i in range(len(self.mature_vector)):
    #         if self.mature_vector!=1:
    #             #this means it is not matured
    #             result=self.find_converging_numbers(self.placeholder[i],threshold)
                
    #             if result!=0:
    #                 self.magnitude_vector[i]=result
    #                 self.mature_vector[i]=1



    def find_converging_numbers(self, float_list, threshold):
        # Check if the list contains less than the threshold number of elements
        if len(float_list) < threshold:
            return 0.0

        # Convert infinities to 0 and keep other numbers as is
        processed_list = [0.0 if math.isinf(num) or math.isnan(num) else num for num in float_list]

        # If all values in the processed list are 0, return 0.0
        if all(num == 0.0 for num in processed_list):
            return 0.0

        # Create a dictionary to count occurrences of each number
        counts = {}
        for num in processed_list:
            # Include numbers within a certain range
            # Adjust this range according to your requirements
            for i in self.float_range(num - 1.0, num + 1.01, 0.01):
                counts[i] = counts.get(i, 0) + 1

        # Find the number with the most converging numbers, ignoring 0
        max_count = 0
        max_num = None
        for num, count in counts.items():
            if count > max_count and num > 0:
                max_count = count
                max_num = num

        # Check if the max count meets the threshold
        if max_count >= threshold and max_num is not None:
            return max_num
        else:
            return 0.0

    def float_range(self, start, stop, step):
        while start < stop:
            yield round(start, 2)  # Round to 2 decimal places, adjust as necessary
            start += step



# def copy_file(source_dir, dest_dir, file_name):
#     source_file = os.path.join(source_dir, file_name)
#     dest_file = os.path.join(dest_dir, file_name)
#     if os.path.exists(source_file):
#         shutil.copy(source_file, dest_file)
#         print(f"File '{file_name}' copied from {source_dir} to {dest_dir}")
#     else:
#         print(f"File '{file_name}' not found in {source_dir}")
#ignore end here

def calculate_median(data):
    num_lists = len(data)
    num_points = len(data[0])

    median_coordinates = []

    for i in range(num_points):
        x_values = []
        y_values = []

        for j in range(num_lists):
            x, y = data[j][i][0]
            if x != 0 and y != 0:  # Ignore zeros as per the instructions
                x_values.append(x)
                y_values.append(y)

        median_x = np.median(x_values) if x_values else 0
        median_y = np.median(y_values) if y_values else 0

        median_coordinates.append([[int(median_x), int(median_y)]])

    return median_coordinates




def merge_lists(list1, list2):
    # Ensure both lists are of the same length
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length")

    # Merge lists index-wise
    merged_list = [list1[i] + list2[i] for i in range(len(list1))]
    return merged_list



class Frame:
    def __init__(self, mean_local, range_local, parts_local, frame_name):
        self.mean_local = mean_local
        self.range_local = range_local
        self.parts_local = parts_local
        self.frame_name=frame_name

    def __eq__(self, other):
        if not isinstance(other, Frame):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return (self.mean_local == other.mean_local and 
                self.range_local == other.range_local and 
                self.parts_local == other.parts_local and
                self.frame_name==frame_name)


def remove_duplicates(frames):
    unique_frames = []
    for frame in frames:
        if frame not in unique_frames:
            unique_frames.append(frame)
    return unique_frames



def merge(merge_parts, merge_file_names1, merge_file_names2):
    result_part=[]
    ranges=[]
    means=[]
    merge_file_names=[]
    for i in merge_parts:
        temp=sorted(i)
        n=len(temp)
        # print(n)
        low_bound=round(0.4*n)
        up_bound=round(0.6*n)
        # print('temp is: ',temp, low_bound,up_bound)
        if n>5:
            range1=temp[up_bound]-temp[low_bound]
            # mean1=statistics.median(temp[low_bound:up_bound])
            mean1=statistics.median(temp)
        elif n==5:
            range1=(temp[3]-temp[1])/2
            mean1=temp[2]
        elif n==4:
            range1=(temp[2]-temp[1])/2
            mean1=(temp[2]+temp[1])/2
        elif n==3:
            range1=0.5
            mean1=temp[1]
        elif n==2:
            range1=0.5
            mean1=(temp[1]+temp[0])/2
        elif n==1:
            range1=0.5
            mean1=temp[0]
        else:
            range1=0.5
            mean1=0
            
        if range1<0.5:
            range1=0.5
        
        if range1>1.5:
            range1=1.5

        result_part.append(temp)
        ranges.append(range1)
        means.append(mean1)
    merge_file_names=merge_file_names1+merge_file_names2

    return Frame(means, ranges, result_part, merge_file_names)





def add_frame(local_database, frame, threshold):
    local_database = remove_duplicates(local_database)

    already_in_local_database=False



    if len(local_database)==0:
        local_database.append(frame)

        for i in local_database:
            print('mean is: ', i.mean_local)
            print('range is: ', i.range_local)
            print('---')


    else:

        means_b=frame.mean_local
        ranges_b=frame.range_local

        for i in range(len(local_database)):
            means_a=local_database[i].mean_local
            ranges_a=local_database[i].range_local
            print(intersect(means_a, means_b, ranges_a, ranges_b, 0.5, map_probs)[0])
            if (intersect(means_a, means_b, ranges_a, ranges_b, 0.5, map_probs)[0]>8.000001e-13):
                print('just after testing intersection function, ',len(local_database))
                print("when merging, find the two vector that are the same.")
                print(means_a)
                print(ranges_a)
                print('--------')
                print(means_b)
                print(ranges_b)

                print('find same one in local database!!')

                merge_parts=merge_lists(local_database[i].parts_local, frame.parts_local)

                frame_final=merge(merge_parts, frame.frame_name, local_database[i].frame_name)
                print('-------------11111111111')
                print(len(local_database))
                print('frame_final', frame_final.mean_local, frame_final.range_local)
                

                local_database[i]=frame_final
                print('successfully merged with same one')
                print(len(local_database))

            else:
                print('add 1 more to local_database')
                # print(len(local_database))
                local_database.append(frame)

                # print(len(local_database))



       

    
    local_database = remove_duplicates(local_database)
    print('--------')
    print('local_database is: ')
    print(len(local_database))
    
    for i in local_database:
        print('mean is: ', i.mean_local)
        print('range is: ', i.range_local)
        print('---')

    return local_database






if __name__ == '__main__':



    #this three line is for the intersection function
    ret_val = dist_intersect(1, 1, 6, 1)
    # print(ret_val)
    map_probs  = gen_dict()  ##get the map of two tailed probabilities

    distri=0.677












    args = get_parser().parse_args()
    # print("args={}".format(args))
    stereo = StereoDepth(args.stereo_file, filter=args.filter)


    # # Directory paths, for 'video'

    # left='./same_3persons/person1front_nojacket/left/'
    # right='./same_3persons/person1front_nojacket/right/'

    # left='./same_3persons/full/left/'
    # right='./same_3persons/full/right/'
    left='./test/left/'
    right='./test/right/'

    # # #for same_person
    left_dir=left
    right_dir=right


    left_files = sorted(os.listdir(left_dir))
    # print(left_files)
    left_files = sorted(left_files, key=sort_numerically)

    #only need one not left and right, have all bbox

    # output_folder='./person1_video3_singleframe/'
    output_folder='./person1231_singleframe'
    # output_folder='./3person_good'


    # #this folder is for good frames

    # result_left_dir = './same_3person_goodframes/left'
    # result_right_dir = './same_3person_goodframes/right'
    result_left_dir=left
    result_right_dir=right

    #how many converge to get the mature sig
    converge_threshold=5
    
    #how many augmentation you wanna do
    num_aug=1
    
    #save top match file or not
    save_top_file=False
    
    #use mean of median
    use_mean=True

    #unit is mm
    effective_range=2643
    # effective_range=2700

    #max min pose threshold
    max_min=0.2

    #get top 3 or ?
    top_ones=11


    person=Person()
    temp_person=[]

    # Process each pair of images

    conf_sum_persons_frames=[]
    conf_sums_with_paths = []


    clean_vector=[]

    good_image_name={}



    #------------------------------------------------------

    frame_pointer=0
    
    #now just consider the local database, the local database is a list of Frame object
    local_database=[]
    local_database_pointer=0
    # recent_five_frame_mean=[]
    # recent_five_frame_range=[]

    recent_five_frames=[]
    recent_five_frame_pointer=0

    #a boolean used to determine when to update the recent five to local database

    mature=False



    amount_threshold=4

    boundary_threshold=3



    left_names=[]

    valid_frames=[]

    threshold=2.000001e-5


    # print(left_files)

    for file_name in left_files:
        # Make sure we're dealing with .jpg files
        # print('-----------------------------------------------------------------------')
        # print('start at frame: ', file_name)
        if file_name.endswith('.jpg'):
            # Construct the corresponding right file name by replacing 'left' with 'right'
            right_file_name = file_name.replace('left', 'right')
            
            left_file_path = os.path.join(left_dir, file_name)
            right_file_path = os.path.join(right_dir, right_file_name)
            
            # Print out the paths to check if they are correct
            # print(f"Checking pair: Left - {left_file_path} | Right - {right_file_path}")

            # Check if the corresponding right file exists
            # print('-------------------------')
            
            # print('start at frame: ', file_name)

            if os.path.exists(right_file_path):
                
                # frame_vectors,fronts,conf_sum_persons = stereo.test_pair_image_file(left_file_path, right_file_path,num_aug)
                
                frameR = cv2.imread(left_file_path)
                frameL = cv2.imread(right_file_path)

                result=stereo.task(frameR, frameL, waitKey=0,show_depth=False, 
                use_augmentation=False, left_file_path=left_file_path, num_aug=num_aug, output_folder=output_folder, pick_biggest_box=True,run_onetime=True,effective_range=effective_range, max_min=max_min)

   
                if result==(None, None, None, None, None):
 
                    # recent_five_frames=[]
                    # recent_five_frame_pointer=0
                    # print('dont clean if there is no frames, clean when it is below threshold')
                    pass

                else:

                    mean_frame, range_frame, part_frame, front, conf_sums_local=result

                    



                    if (mean_frame!=None)and(range_frame!=None):
                        
                        # print(mean_frame, range_frame, part_frame)

                        left_names.append(file_name)


                        this_frame=Frame(mean_frame, range_frame, part_frame, file_name)

                        valid_frames.append(this_frame)


















