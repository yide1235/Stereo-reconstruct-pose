

#include "ImageCompare.hpp"
#include <cmath> // For sqrt
#include <iostream> // For std::cerr
#include <cstring> // For std::memcpy




// //image compare class:
// class ImageCompare {


// private:
//     cv::Mat imgL;
//     cv::Mat imgR;


// public:
//     // ImageCompare(std::unique_ptr<tflite::Interpreter> interpreter, cv::Mat left, cv::Mat right){
//     ImageCompare( cv::Mat left, cv::Mat right){
//         imgL=left;
//         imgR=right;
//     }


//     cv::Mat load_image(const std::string& img_path, const cv::Size& size = cv::Size(224, 224)) {
//         cv::Mat img = cv::imread(img_path);
//         cv::resize(img, img, size);
//         cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//         img.convertTo(img, CV_32F, 1.0 / 127.5, -1.0);
//         return img;
//     }
    

//     cv::Mat recompute_image(const cv::Mat& img, int x, int y, int box_szx, int box_szy, int model_size) {
//         int x1 = x - box_szx / 2;
//         int y1 = y - box_szy / 2;
//         cv::Rect roi(x1, y1, box_szx, box_szy);
//         cv::Mat deep_img=img.clone();
//         cv::Mat cropped_img = deep_img(roi);
//         if (model_size > 0) {
//             cv::resize(cropped_img, cropped_img, cv::Size(model_size, model_size));
//         }
//         cv::cvtColor(cropped_img, cropped_img, cv::COLOR_BGR2RGB);
//         cropped_img.convertTo(cropped_img, CV_32F);
//         cropped_img = cropped_img / 127.5 - 1.0;
//         return cropped_img;
//     }



//     cv::Mat get_image(const cv::Mat& image1, int x1, int y1, int width1, int height1, int model_size) {
//         cv::Rect roi(x1, y1, width1, height1);
//         cv::Mat img = image1(roi);
//         if (model_size > 0) {
//             cv::resize(img, img, cv::Size(model_size, model_size));
//         }
//         cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//         img.convertTo(img, CV_32F, 1.0 / 127.5, -1.0);
//         return img;
//     }

//     float calculateNormManually(const cv::Mat& vec) {
//         if (vec.rows != 1 && vec.cols != 1) {
//             std::cerr << "The input should be a single row or column matrix." << std::endl;
//             return -1;
//         }
//         double sum = 0.0;
//         if (vec.rows == 1) {
//             for (int i = 0; i < vec.cols; ++i) {
//                 float value = vec.at<float>(0, i);
//                 sum += value * value;
//             }
//         } else {
//             for (int i = 0; i < vec.rows; ++i) {
//                 float value = vec.at<float>(i, 0);
//                 sum += value * value;
//             }
//         }
//         return std::sqrt(sum);
//     }


//     float cosine_similarity(const cv::Mat& vec1, const cv::Mat& vec2) {
//         float dot = vec1.dot(vec2);
//         float denom = norm(vec1) * norm(vec2);
//         return static_cast<float>(dot / denom);
//     }


//     float euclidean_distance(const cv::Mat& vec1, const cv::Mat& vec2) {
//         cv::Mat diff = vec1 - vec2;
//         return static_cast<float>(norm(diff));
//     }


//     float abs_diff_image(const cv::Mat& img1, int x1, int y1, const cv::Mat& img2, int x2, int y2) {
//         cv::Mat arr1 = get_image(img1, x1-1, y1-1, 3, 3, -1); // Using -1 for `model_size` to indicate no resizing
//         cv::Mat arr2 = get_image(img2, x2-1, y2-1, 3, 3, -1);
//         return euclidean_distance(arr1, arr2);
//     }

//     cv::Mat runTFLiteModel(const std::unique_ptr<tflite::Interpreter>& interpreter, const cv::Mat& image) {
//         // cv::Mat input_image;
//         // // image.convertTo(input_image, CV_32F); // Convert image to float
//         // if(image.type() != CV_32F) {
//         //     image.convertTo(input_image, CV_32F); 
//         // }
//         cv::Mat input_image=image;
//         TfLiteTensor* input_tensor = interpreter->tensor(interpreter->inputs()[0]);
//         const uint HEIGHT = input_tensor->dims->data[1];
//         const uint WIDTH = input_tensor->dims->data[2];
//         const uint CHANNEL = input_tensor->dims->data[3];
//         std::memcpy(input_tensor->data.f, input_image.ptr<float>(0), HEIGHT*WIDTH*CHANNEL* sizeof(float));
//         interpreter->Invoke();
//         // auto end = std::chrono::high_resolution_clock::now();
//         // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//         // std::cout << "Time taken for recompute_image: " << duration << " milliseconds" << std::endl;
//         const TfLiteTensor* output_tensor = interpreter->tensor(interpreter->outputs()[0]);
//         const float* output_data = output_tensor->data.f;
//         int output_size = output_tensor->dims->data[output_tensor->dims->size - 1];
//         cv::Mat feature_vector(1, output_size, CV_32F, const_cast<float*>(output_data));
//         return feature_vector.clone(); // Return a clone to ensure data ownership by cv::Mat

//     }


//     bool areMatricesEqualWithinTolerance(const cv::Mat& mat1, const cv::Mat& mat2, double tol = 1e-5) {
//         if (mat1.type() != mat2.type() || mat1.size() != mat2.size()) {
//             return false;
//         }
//         cv::Mat diff;
//         cv::absdiff(mat1, mat2, diff); // Compute the absolute difference.
//         diff = diff > tol; // Check if difference is greater than tolerance.
//         return cv::countNonZero(diff) == 0; // If no elements are above tolerance, matrices are considered equal.
//     }

//     std::pair<int, int> search_image(const std::unique_ptr<tflite::Interpreter>& interpreter,  int left_x, int left_y, int box_sz, int model_sz){
//         cv::Mat refImg = recompute_image(imgL, left_x, left_y, box_sz, box_sz, model_sz); 
//         cv::Mat featureVec1 = runTFLiteModel(interpreter, refImg);
//         int refX = left_x;
//         const int delta = 10; // Step size for sliding window
//         float lastSimilarity = 0.0;
//         int maxX = 0;
//         float minSlope = std::numeric_limits<float>::max();
//         int maxX2 = 0;
//         int cnt = 0;
//         std::vector<std::pair<float, int>> deltaSim;
//         while (true) {
//             if (lastSimilarity == 0.0){
//                 cv::Mat cmpImg = recompute_image(imgR, refX, left_y, box_sz, box_sz, model_sz);
//                 cv::Mat featureVec2 = runTFLiteModel(interpreter, cmpImg);
//                 lastSimilarity=cosine_similarity(featureVec1, featureVec2);
//                 // std::cout << lastSimilarity<< std::endl;
//             }
//             cv::Mat cmpImg = recompute_image(imgR, refX - ((cnt + 1) * delta), left_y, box_sz, box_sz, model_sz);
//             cv::Mat featureVec2 = runTFLiteModel(interpreter,cmpImg);
//             float similarity1 = cosine_similarity(featureVec1, featureVec2);
//             // std::cout << similarity1 << std::endl;
//             deltaSim.push_back(std::make_pair(similarity1 - lastSimilarity, refX - ((cnt + 1) * delta)));
//             if (std::abs(similarity1 - lastSimilarity) < minSlope) {
//                 maxX2 = maxX;
//                 maxX = refX - ((cnt + 1) * delta);
//                 minSlope = std::abs(similarity1 - lastSimilarity);
//             }
//             if (similarity1 - lastSimilarity < 0) {
//                 break; // Assuming crossing a local maximum
//             } else {
//                 lastSimilarity = similarity1;
//             }
//             cnt++;
//         }
//         int mult = (maxX2 > maxX) ? -1 : 1;
//         int optX = maxX;
//         float minDiffR = std::numeric_limits<float>::max();
//         bool maxSet = false;
//         for (int sch = 0; sch < int(delta / 2); ++sch) {
//             float diffR = abs_diff_image(imgR, maxX + (sch * mult), left_y, imgL, left_x, left_y);
//             // std::cout << diffR<< std::endl;
//             cv::Mat cmpImg = recompute_image(imgR, maxX + (sch * mult), left_y, box_sz, box_sz, model_sz);
//             cv::Mat featureVec2 = runTFLiteModel(interpreter,cmpImg);
//             float similarity = cosine_similarity(featureVec1, featureVec2);

//             float totalDiff = similarity - diffR;
//             if (totalDiff > minDiffR) {
//                 if (minDiffR != std::numeric_limits<float>::max()) {
//                     maxSet = true;
//                 }
//                 minDiffR = totalDiff;
//                 optX = maxX + (sch * mult);
//             } else if (maxSet) {
//                 break;
//             }
//         }
//         // return optX;
//         return std::make_pair(optX, left_y);
//     }
// };


ImageCompare::ImageCompare(cv::Mat left, cv::Mat right) : imgL(left.clone()), imgR(right.clone()) {}


// Load an image from a file, resize it, convert to RGB and normalize
cv::Mat ImageCompare::load_image(const std::string& img_path, const cv::Size& size) {
    cv::Mat img = cv::imread(img_path);
    cv::resize(img, img, size);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 127.5, -1.0);
    return img;
}

// Recompute an image segment based on the given parameters
cv::Mat ImageCompare::recompute_image(const cv::Mat& img, int x, int y, int box_szx, int box_szy, int model_size) {
    int x1 = x - box_szx / 2;
    int y1 = y - box_szy / 2;
    cv::Rect roi(x1, y1, box_szx, box_szy);
    cv::Mat cropped_img = img(roi).clone();
    if (model_size > 0) {
        cv::resize(cropped_img, cropped_img, cv::Size(model_size, model_size));
    }
    cv::cvtColor(cropped_img, cropped_img, cv::COLOR_BGR2RGB);
    cropped_img.convertTo(cropped_img, CV_32F, 1.0 / 127.5, -1.0);
    return cropped_img;
}

cv::Mat ImageCompare::get_image(const cv::Mat& image1, int x1, int y1, int width1, int height1, int model_size) {
    cv::Rect roi(x1, y1, width1, height1);
    cv::Mat img = image1(roi);
    if (model_size > 0) {
        cv::resize(img, img, cv::Size(model_size, model_size));
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 127.5, -1.0);
    return img;
}


float ImageCompare::calculateNormManually(const cv::Mat& vec) {
    if (vec.rows != 1 && vec.cols != 1) {
        std::cerr << "The input should be a single row or column matrix." << std::endl;
        return -1;
    }
    double sum = 0.0;
    if (vec.rows == 1) {
        for (int i = 0; i < vec.cols; ++i) {
            float value = vec.at<float>(0, i);
            sum += value * value;
        }
    } else {
        for (int i = 0; i < vec.rows; ++i) {
            float value = vec.at<float>(i, 0);
            sum += value * value;
        }
    }
    return std::sqrt(sum);
}


float ImageCompare::cosine_similarity(const cv::Mat& vec1, const cv::Mat& vec2) {
    float dot = vec1.dot(vec2);
    float denom = norm(vec1) * norm(vec2);
    return static_cast<float>(dot / denom);
}


float ImageCompare::euclidean_distance(const cv::Mat& vec1, const cv::Mat& vec2) {
    cv::Mat diff = vec1 - vec2;
    return static_cast<float>(norm(diff));
}


float ImageCompare::abs_diff_image(const cv::Mat& img1, int x1, int y1, const cv::Mat& img2, int x2, int y2) {
    cv::Mat arr1 = get_image(img1, x1-1, y1-1, 3, 3, -1); // Using -1 for `model_size` to indicate no resizing
    cv::Mat arr2 = get_image(img2, x2-1, y2-1, 3, 3, -1);
    return euclidean_distance(arr1, arr2);
}


cv::Mat ImageCompare::runTFLiteModel(const std::unique_ptr<tflite::Interpreter>& interpreter, const cv::Mat& image) {
    // cv::Mat input_image;
    // // image.convertTo(input_image, CV_32F); // Convert image to float
    // if(image.type() != CV_32F) {
    //     image.convertTo(input_image, CV_32F); 
    // }
    cv::Mat input_image=image;
    TfLiteTensor* input_tensor = interpreter->tensor(interpreter->inputs()[0]);
    const uint HEIGHT = input_tensor->dims->data[1];
    const uint WIDTH = input_tensor->dims->data[2];
    const uint CHANNEL = input_tensor->dims->data[3];
    std::memcpy(input_tensor->data.f, input_image.ptr<float>(0), HEIGHT*WIDTH*CHANNEL* sizeof(float));
    interpreter->Invoke();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "Time taken for recompute_image: " << duration << " milliseconds" << std::endl;
    const TfLiteTensor* output_tensor = interpreter->tensor(interpreter->outputs()[0]);
    const float* output_data = output_tensor->data.f;
    int output_size = output_tensor->dims->data[output_tensor->dims->size - 1];
    cv::Mat feature_vector(1, output_size, CV_32F, const_cast<float*>(output_data));
    return feature_vector.clone(); // Return a clone to ensure data ownership by cv::Mat

}


bool ImageCompare::areMatricesEqualWithinTolerance(const cv::Mat& mat1, const cv::Mat& mat2, double tol) {
    if (mat1.type() != mat2.type() || mat1.size() != mat2.size()) {
        return false;
    }
    cv::Mat diff;
    cv::absdiff(mat1, mat2, diff); // Compute the absolute difference.
    diff = diff > tol; // Check if difference is greater than tolerance.
    return cv::countNonZero(diff) == 0; // If no elements are above tolerance, matrices are considered equal.
}


std::pair<int, int> ImageCompare::search_image(const std::unique_ptr<tflite::Interpreter>& interpreter,  int left_x, int left_y, int box_sz, int model_sz){
    cv::Mat refImg = recompute_image(imgL, left_x, left_y, box_sz, box_sz, model_sz); 
    cv::Mat featureVec1 = runTFLiteModel(interpreter, refImg);
    int refX = left_x;
    const int delta = 10; // Step size for sliding window
    float lastSimilarity = 0.0;
    int maxX = 0;
    float minSlope = std::numeric_limits<float>::max();
    int maxX2 = 0;
    int cnt = 0;
    std::vector<std::pair<float, int>> deltaSim;
    while (true) {
        if (lastSimilarity == 0.0){
            cv::Mat cmpImg = recompute_image(imgR, refX, left_y, box_sz, box_sz, model_sz);
            cv::Mat featureVec2 = runTFLiteModel(interpreter, cmpImg);
            lastSimilarity=cosine_similarity(featureVec1, featureVec2);
            // std::cout << lastSimilarity<< std::endl;
        }
        cv::Mat cmpImg = recompute_image(imgR, refX - ((cnt + 1) * delta), left_y, box_sz, box_sz, model_sz);
        cv::Mat featureVec2 = runTFLiteModel(interpreter,cmpImg);
        float similarity1 = cosine_similarity(featureVec1, featureVec2);
        // std::cout << similarity1 << std::endl;
        deltaSim.push_back(std::make_pair(similarity1 - lastSimilarity, refX - ((cnt + 1) * delta)));
        if (std::abs(similarity1 - lastSimilarity) < minSlope) {
            maxX2 = maxX;
            maxX = refX - ((cnt + 1) * delta);
            minSlope = std::abs(similarity1 - lastSimilarity);
        }
        if (similarity1 - lastSimilarity < 0) {
            break; // Assuming crossing a local maximum
        } else {
            lastSimilarity = similarity1;
        }
        cnt++;
    }
    int mult = (maxX2 > maxX) ? -1 : 1;
    int optX = maxX;
    float minDiffR = std::numeric_limits<float>::max();
    bool maxSet = false;
    for (int sch = 0; sch < int(delta / 2); ++sch) {
        float diffR = abs_diff_image(imgR, maxX + (sch * mult), left_y, imgL, left_x, left_y);
        // std::cout << diffR<< std::endl;
        cv::Mat cmpImg = recompute_image(imgR, maxX + (sch * mult), left_y, box_sz, box_sz, model_sz);
        cv::Mat featureVec2 = runTFLiteModel(interpreter,cmpImg);
        float similarity = cosine_similarity(featureVec1, featureVec2);

        float totalDiff = similarity - diffR;
        if (totalDiff > minDiffR) {
            if (minDiffR != std::numeric_limits<float>::max()) {
                maxSet = true;
            }
            minDiffR = totalDiff;
            optX = maxX + (sch * mult);
        } else if (maxSet) {
            break;
        }
    }
    // return optX;
    return std::make_pair(optX, left_y);
}












// //main function
// int main() {

//     // auto start_time = std::chrono::high_resolution_clock::now();

//     // Load images
//     cv::Mat left = cv::imread("../feature_extractor/left_front5.jpg");
//     cv::Mat right = cv::imread("../feature_extractor/right_front5.jpg");

//     // // create model for feature extraction, this piece of code cannot make into functions.
//     std::unique_ptr<tflite::FlatBufferModel> model =
//         tflite::FlatBufferModel::BuildFromFile("../feature_extractor/mobilenetv2_quant_int8.tflite");

//     //   auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
//     //   auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     std::unique_ptr<tflite::Interpreter> interpreter;
//     tflite::InterpreterBuilder(*model, resolver)(&interpreter);
//     //   interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);
//     interpreter->SetAllowFp16PrecisionForFp32(false);
//     interpreter->AllocateTensors();



//     // Create ImageCompare instance
//     ImageCompare cmp(left, right);

//     // int optx = cmp.search_image(interpreter, "front5.jpg", 770,376,224, 224);
//     std::pair<int, int> result_point = cmp.search_image(interpreter,  844,387,224, 224);

//     std::cout << result_point.first << " "<< result_point.second << std::endl;

//     // auto end_time = std::chrono::high_resolution_clock::now();
//     // auto time_taken_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
//     // std::cout << "Optimal X (first search): " << optx << std::endl;
//     // std::cout << "Time taken (first search): " << time_taken_ms << " ms" << std::endl;
//     return 0;
// }
