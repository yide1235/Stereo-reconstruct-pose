// detection.hpp
#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <vector>
#include <string>

using namespace std;
using namespace cv;


// Declare the functions
cv::Mat mat_process(cv::Mat src, uint width, uint height);
cv::Mat letterbox(cv::Mat img, int height, int width);
void setupInput(const std::unique_ptr<tflite::Interpreter>& interpreter);
std::vector<float> xywh2xyxy_scale(const std::vector<float>& boxes, float width, float height);
std::vector<float> scaleBox(const std::vector<float>& box, int img1Height, int img1Width, int img0Height, int img0Width);
std::vector<int> NMS(const std::vector<std::vector<float>>& boxes, float overlapThresh);
void expandBoundingBoxes(std::vector<std::vector<float>>& boxes, int img_width, int img_height, float width_add, float height_add);
std::vector<std::vector<float>> detection_process(const std::unique_ptr<tflite::Interpreter>& interpreter,const cv::Mat& img, const float detection_threshold);
cv::Scalar hex2rgb(const std::string& h);
cv::Scalar getColor(int i, bool bgr = false);
::Mat plotOneBox(const std::vector<float>& x, cv::Mat im, cv::Scalar color = cv::Scalar(128, 128, 128), const std::string& label = "", int rectLineThickness = 3, int textLineThickness = 2);
void plotBboxes(const cv::Mat& img, const std::vector<std::vector<float>>& results, const std::vector<std::string>& coco_names, const std::string& savePath);

#endif // DETECTION_HPP