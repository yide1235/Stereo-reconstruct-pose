// triangulation.hpp
#ifndef TRIANGULATION_HPP
#define TRIANGULATION_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <string>
#include <vector>
#include <map>
#include "common_types.hpp"


bool load_stereo_coefficients(const std::string &filename, cv::Mat &K1, cv::Mat &D1, cv::Mat &K2, cv::Mat &D2, cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F, cv::Size &imageSize);

std::map<std::string, cv::Mat> get_stereo_coefficients(const std::string &stereo_file, bool rectify = true);

cv::Mat get_3dpoints(const cv::Mat& disparity, const cv::Mat& Q, float scale = 1.0f);

std::pair<cv::Mat, cv::Mat> get_rectify_image(const cv::Mat &imgL, const cv::Mat &imgR, const std::map<std::string, cv::Mat> &camera_config);

cv::Mat get_filter_disparity(cv::Mat& imgL, cv::Mat& imgR, bool use_wls = true);

void printKeypoints(const std::vector<Keypoint>& keypoints);

void printDepthValues(const cv::Mat& depth, int numRowsToPrint = 5, int numColsToPrint = 5);

std::vector<float> calculateDistances(const std::vector<cv::Vec3f>& depth_3d, const std::vector<std::vector<int>>& vec_inds);

void printDepth3D(const std::vector<cv::Vec3f>& depth_3d);

#endif // TRIANGULATION_HPP

