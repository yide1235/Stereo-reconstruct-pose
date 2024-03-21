

#include "ImageCompare.hpp"
#include <cmath> // For sqrt
#include <iostream> // For std::cerr
#include <cstring> // For std::memcpy


ImageCompare::ImageCompare(cv::Mat left, cv::Mat right) : imgL(left.clone()), imgR(right.clone()) {}


// // Load an image from a file, resize it, convert to RGB and normalize
// cv::Mat ImageCompare::load_image(const std::string& img_path, const cv::Size& size) {
//     cv::Mat img = cv::imread(img_path);
//     cv::resize(img, img, size);
//     cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//     img.convertTo(img, CV_32F, 1.0 / 127.5, -1.0);
//     return img;
// }

// Recompute an image segment based on the given parameters
cv::Mat ImageCompare::recompute_image(const cv::Mat& img, int x, int y, int box_szx, int box_szy, int model_size) {
    int x1 = x - static_cast<int>(box_szx / 2);
    int y1 = y - static_cast<int>(box_szy / 2);

    cv::Mat refimg= get_image(img, x-x1, y-y1, box_szx, box_szy, model_size);

    return refimg;
}

cv::Mat ImageCompare::get_image(const cv::Mat& image1, int x1, int y1, int width1, int height1, int model_size) {
    cv::Rect roi(x1, y1, width1, height1);
    cv::Mat img = image1(roi);
    // if (model_size > 0) {
    //     cv::resize(img, img, cv::Size(model_size, model_size));
    // }
    if (model_size > 0 && (width1 != height1 || width1 != model_size)) {
        std::cout << "resizing" << std::endl;
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


std::pair<int, int> ImageCompare::search_image(const std::unique_ptr<tflite::Interpreter>& interpreter,  int x, int y, int box_sz, int model_size){
    
    cv::Mat refImg = recompute_image(imgL, x, y, box_sz, box_sz, model_size); 
    cv::Mat feature_vec1 = runTFLiteModel(interpreter, refImg);
    int refx = x;
    const int delta = 10; // Step size for sliding window
    float last_similarity = 0.0;
    int maxx = x;
    float minslope = 1;
    int maxx2 = x;
    int cnt = 0;
    int tmaxx=x;

    int mult;

    std::vector<std::pair<float, int>> delta_sim;

    while (true) {
        if (last_similarity == 0.0){
            cv::Mat cmpimg = recompute_image(imgR, refx, y, box_sz, box_sz, model_size);
            cv::Mat feature_vec2 = runTFLiteModel(interpreter, cmpimg);

            float diffr=abs_diff_image(imgR, refx, y, imgL, x, y);
            last_similarity=cosine_similarity(feature_vec1, feature_vec2)-diffr;
            // std::cout << lastSimilarity<< std::endl;
        }
        cv::Mat cmpimg = recompute_image(imgR, refx + ((cnt + 1) * delta), y, box_sz, box_sz, model_size);
        cv::Mat feature_vec2 = runTFLiteModel(interpreter,cmpimg);

        float diffr1=abs_diff_image(imgR, refx + ((cnt + 1) * delta), y, imgL, x, y);

        float similarity1 = cosine_similarity(feature_vec1, feature_vec2)-diffr1;
        // std::cout << "curr similarity is: " << similarity1 << "location: "<< refx + ((cnt + 1) * delta) << std::endl;

        // std::cout << "similarity: " << similarity1 - last_similarity << std::endl;


        delta_sim.push_back(std::make_pair(similarity1 - last_similarity, refx + ((cnt + 1) * delta)));

        if(similarity1>last_similarity){
            tmaxx=refx + ((cnt + 1) * delta);
        }


        if (std::abs(similarity1 - last_similarity) < minslope) {
            maxx2 = maxx;
            maxx = refx + ((cnt + 1) * delta);
            minslope = std::abs(similarity1 - last_similarity);

            // std::cout << "set maxx : " << maxx <<"min slope : " << minslope << std::endl;

        }
        if (similarity1 - last_similarity < 0) {
            break; // Assuming crossing a local maximum
        } else {
            last_similarity = similarity1;
        }
        cnt++;
    }


    // std::cout << last_similarity << std::endl;


    // int mult = (maxX2 > maxX) ? -1 : 1;
    if(tmaxx>maxx){
        mult=-1;
        maxx=tmaxx;

    }else{
        mult=1;
        maxx=tmaxx;
    }


    int optx = maxx;

    float mindiffr = -1;
    bool maxset = false;

    int running_average=0;
    int avg_count=0;

    for (int sch = 0; sch < int(delta)+1; ++sch) {
        float diffr = abs_diff_image(imgR, maxx + (sch * mult), y, imgL, x, y);
        // std::cout << diffR<< std::endl;
        cv::Mat cmpimg = recompute_image(imgR, maxx + (sch * mult), y, box_sz, box_sz, model_size);
        cv::Mat feature_vec2 = runTFLiteModel(interpreter,cmpimg);
        float similarity = cosine_similarity(feature_vec1, feature_vec2);

        float totdiff = similarity - diffr;
        // std::cout << maxx+(sch*mult) << "totdiff"<< totdiff << "similarity" << similarity << "diffr"<< diffr << std::endl;



        running_average=running_average+totdiff;
        avg_count+=1;

        // std::cout << "running average is: " << running_average/avg_count << "count is: " << avg_count << std::endl;


        if (totdiff > mindiffr) {
            if (mindiffr != -1) {
                maxset = true;
            }
            mindiffr = totdiff;
            optx = maxx + (sch * mult);
        } 

        if ((avg_count>3)&&(totdiff< running_average/avg_count)){
            break;
        }
    }
    // return optX;
    return std::make_pair(optx, y);
}





// //main function
// int main() {

//     // auto start_time = std::chrono::high_resolution_clock::now();

//     // Load images
//     cv::Mat left = cv::imread("../deep_ssim/rectL.jpg");
//     cv::Mat right = cv::imread("../deep_ssim/rectR.jpg");

//     // // create model for feature extraction, this piece of code cannot make into functions.
//     std::unique_ptr<tflite::FlatBufferModel> model =
//         tflite::FlatBufferModel::BuildFromFile("../deep_ssim/mobilenetv2_quant_int8.tflite");

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

//     // right should be 920 256 or 919 256
//     std::pair<int, int> result_point = cmp.search_image(interpreter,  975,266,224, 224);

//     std::cout << result_point.first << " "<< result_point.second << std::endl;

//     // result_point = cmp.search_image(interpreter,  975,266,224, 224);

//     // std::cout << result_point.first << " "<< result_point.second << std::endl;



//     // result_point = cmp.search_image(interpreter,  1016,411,224, 224);

//     // std::cout << result_point.first << " "<< result_point.second << std::endl;

//     // auto end_time = std::chrono::high_resolution_clock::now();
//     // auto time_taken_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
//     // std::cout << "Optimal X (first search): " << optx << std::endl;
//     // std::cout << "Time taken (first search): " << time_taken_ms << " ms" << std::endl;
//     return 0;
// }





// cmp = image_compare('mobilenetv2_quant_int8.tflite')

// optx = cmp.search_image("front5.png",900,256, 224, 224) ###---> 920, 256
// print(optx)
// print('---------')

// # optx = cmp.search_image("front5.png",975,266, 224, 224) ###---> 993, 266
// # print(optx)
// # print('---------')

// # optx = cmp.search_image("front5.png",715,550, 224, 224) ###---> 730, 550
// # print(optx)
// # print('---------')

// # optx = cmp.search_image("front5.png",940,521, 224, 224) ###---> 955, 521
// # print(optx)
// # print('---------')


// # optx = cmp.search_image("front5.png",997,496, 224, 224) ###---> 1012, 496
// # print(optx)
// # print('---------')


// # optx = cmp.search_image("front5.png",1016,411, 224, 224) ###---> 1032, 411
// # print(optx)
// # print('---------')


