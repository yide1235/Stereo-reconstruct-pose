#ifndef IMAGE_COMPARE_HPP
#define IMAGE_COMPARE_HPP

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <utility> // For std::pair

class ImageCompare {
private:
    cv::Mat imgL;
    cv::Mat imgR;

public:
    ImageCompare(cv::Mat left, cv::Mat right);

    cv::Mat load_image(const std::string& img_path, const cv::Size& size = cv::Size(224, 224));
    cv::Mat recompute_image(const cv::Mat& img, int x, int y, int box_szx, int box_szy, int model_size);
    cv::Mat get_image(const cv::Mat& image1, int x1, int y1, int width1, int height1, int model_size);
    float calculateNormManually(const cv::Mat& vec);
    float cosine_similarity(const cv::Mat& vec1, const cv::Mat& vec2);
    float euclidean_distance(const cv::Mat& vec1, const cv::Mat& vec2);
    float abs_diff_image(const cv::Mat& img1, int x1, int y1, const cv::Mat& img2, int x2, int y2);
    cv::Mat runTFLiteModel(const std::unique_ptr<tflite::Interpreter>& interpreter, const cv::Mat& image);
    bool areMatricesEqualWithinTolerance(const cv::Mat& mat1, const cv::Mat& mat2, double tol = 1e-5);
    std::pair<int, int> search_image(const std::unique_ptr<tflite::Interpreter>& interpreter, int left_x, int left_y, int box_sz, int model_sz);
};

#endif // IMAGE_COMPARE_HPP

