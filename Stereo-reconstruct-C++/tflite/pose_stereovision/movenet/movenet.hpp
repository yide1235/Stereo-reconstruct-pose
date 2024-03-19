// movenet.hpp
#ifndef MOVENET_HPP
#define MOVENET_HPP

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <vector>
#include <string>
#include <random>
#include "../lib/common_types.hpp"

using namespace std;
using namespace cv;

typedef cv::Point3_<float> Pixel;



// Keypoint structure

// Function declarations
cv::Mat ResizeWithPad(const cv::Mat& src, int target_width, int target_height);
std::vector<Keypoint> process_movenet(const std::unique_ptr<tflite::Interpreter>& interpreter, const cv::Mat& img, float movenet_threshold);
std::vector<std::vector<Keypoint>> process_movenet_augmentation(const std::unique_ptr<tflite::Interpreter>& interpreter, const cv::Mat& img, float movenet_threshold, int loop_threshold, bool use_aug=true);
std::vector<Keypoint> process_movenet_onetime(const std::unique_ptr<tflite::Interpreter>& interpreter, const cv::Mat& img, float movenet_threshold, int loop_threshold);
void drawLinesBetweenPoints(cv::Mat& image, const std::vector<Keypoint>& keypoints, const std::vector<std::vector<int>>& vec_inds);

#endif // MOVENET_HPP