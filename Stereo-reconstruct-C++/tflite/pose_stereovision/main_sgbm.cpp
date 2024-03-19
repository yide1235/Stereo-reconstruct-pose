#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/model.h"
#include <vector>
#include <cmath>
#include <map>
#include <string>
#include <random>
#include <filesystem> 
#include <iostream>
#include <algorithm>
#include <numeric>
#include <regex>
#include <utility> 

#include "opencv481.h"

#include "lib/intersection.hpp"
#include "detection/detection.hpp"
#include "movenet/movenet.hpp"
#include "lib/triangulation.hpp"
#include "lib/post_processing.hpp"
#include "deep_ssim/ImageCompare.hpp"

using namespace std;
using namespace cv;
using namespace tflite;
namespace fs = std::filesystem;




//main function
//****************************************
Frame process_eachframe(const std::unique_ptr<tflite::Interpreter>& detection_interpreter, const std::unique_ptr<tflite::Interpreter>& movenet_interpreter,  const std::string& imgf1,  const std::string& imgf2, const std::string& output_folder)
{


    float movenet_threshold=0.3;
    float detection_threshold=0.57;
    int loop_theshold=8;
    float variance_threshold=3;
    int required_variance_point=9;
    // double intersect_threshold=2.000001e-7;
    int effective_range=2737;


    std::vector<std::string> coco_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    };


    cv::Mat img1 = cv::imread(imgf1);
    if (img1.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        // You should return an empty cv::Mat or handle errors differently.
        return Frame();
    }

    cv::Mat img2 = cv::imread(imgf2);
    if (img2.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        // You should return an empty cv::Mat or handle errors differently.
        return Frame();
    }

    std::string stereo_file = "../cam_config/stereo_cam.yml";

    // Call the function and get the camera configuration
    std::map<std::string, cv::Mat> camera_config = get_stereo_coefficients();

    cv::Mat imgL = cv::imread(imgf1);
    cv::Mat imgR = cv::imread(imgf2);


    // Rectify the images
    auto [rectifiedL, rectifiedR] = get_rectify_image(imgL, imgR, camera_config);


    std::vector<std::vector<float>> results1 = detection_process(detection_interpreter,rectifiedL, detection_threshold);


    //*****************************************
    //start to get the 3d depth

    cv::Mat grayL, grayR;
    cv::cvtColor(rectifiedL, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rectifiedR, grayR, cv::COLOR_BGR2GRAY);

    cv::Mat grayL_copy=grayL.clone();
    cv::Mat grayR_copy=grayR.clone();


    cv::Mat dispL = get_filter_disparity(grayL_copy, grayR_copy, true);


    // ***** the disparity is correct

    // Convert the disparity map to 3D points
    cv::Mat Q = camera_config["Q"];
    cv::Mat points_3d = get_3dpoints(dispL, Q);


    if (points_3d.empty()) {
        std::cerr << "Error: points_3d matrix is empty." << std::endl;
        return Frame();
    }

    if (points_3d.type() != CV_32FC3) {
        std::cerr << "Error: points_3d matrix is not of type CV_32FC3." << std::endl;
        return Frame();
    }

    if (points_3d.channels() != 3) {
        std::cerr << "Error: points_3d does not have 3 channels." << std::endl;
        return Frame();
    }

    std::vector<cv::Mat> channels;
    cv::split(points_3d, channels);
    cv::Mat x = channels[0];
    cv::Mat y = channels[1];
    cv::Mat depth = channels[2];


    // xyz_coord is the same as points_3d
    cv::Mat xyz_coord = points_3d; 



    //*****************************************
    //this is for movenet

    std::vector<std::vector<int>> vec_inds = {
        {6, 5}, {6, 8}, {8, 10}, {5, 7}, {7, 9}, {12, 14}, {14, 16}, {11, 13}, {13, 15}, {6, 12}, {5, 11}, {12, 11}
    };

    std::vector<float> box1;

    box1=findLargestBBox(results1);

    int x1 = std::max(0, static_cast<int>(box1[0] - 0.05 * (box1[2] - box1[0])));
    int y1 = std::max(0, static_cast<int>(box1[1] - 0.05 * (box1[3] - box1[1])));
    int x2 = std::min(rectifiedL.cols, static_cast<int>(box1[2] + 0.05 * (box1[2] - box1[0])));
    int y2 = std::min(rectifiedL.rows, static_cast<int>(box1[3] + 0.05 * (box1[3] - box1[1])));

    // Crop and save the image
    cv::Rect cropRect(x1, y1, x2 - x1, y2 - y1);
    cv::Mat crop1 = rectifiedL(cropRect);


    std::vector<std::vector<Keypoint>> left=process_movenet_augmentation(movenet_interpreter, crop1, movenet_threshold, loop_theshold, true);

    std::vector<std::vector<float>> list_of_mag;

    cv::Mat rectifiedL_copy=rectifiedL.clone();

    std::vector<float> right_shoulder;

    for (int i=0;i< left.size();i++){


        std::vector<Keypoint> left_converted;

        for (const auto& point : left[i]) {
            // cv::Point center(static_cast<int>(point.x), static_cast<int>(point.y));
            // cv::circle(crop1, center, 2, cv::Scalar(0, 255, 0), -1);
            float x_adj = point.x + std::max(0, static_cast<int>(box1[0] - 0.05 * (box1[2] - box1[0])));
            float y_adj = point.y + std::max(0, static_cast<int>(box1[1] - 0.05 * (box1[3] - box1[1])));


            Keypoint temp={x_adj, y_adj};
            left_converted.push_back(temp);
        }


        //end of movenet

        //*****************************************
        //start to use the result of movenet to get the 3d point


        std::vector<cv::Vec3f> depth_3d;

        for (const auto& keypoint : left_converted) {
            int tempx = static_cast<int>(keypoint.x);
            int tempy = static_cast<int>(keypoint.y);

            

            // Clamp tempx and tempy to the valid range of the points_3d matrix
            tempx = std::max(0, std::min(tempx, points_3d.cols - 1));
            tempy = std::max(0, std::min(tempy, points_3d.rows - 1));

            
            // Access the 3D point at (tempx, tempy)
            cv::Vec3f point3d = points_3d.at<cv::Vec3f>(tempy, tempx);

            // Store the 3D point
            depth_3d.push_back(point3d);
        }

        cv::Vec3f sixthElement = depth_3d[5];

        // Access the third component (index 2) of the cv::Vec3f
        float thirdComponent = sixthElement[2];

        right_shoulder.push_back(thirdComponent);

        std::vector<float> distances = calculateDistances(depth_3d, vec_inds);

        list_of_mag.push_back(distances);



    }

    //test effective range
    float sum = std::accumulate(right_shoulder.begin(), right_shoulder.end(), 0.0f);
    float average = 0.0f;

    if (!right_shoulder.empty()) {
        average = sum / static_cast<float>(right_shoulder.size());
    }

    // std::cout << average << "--" << effective_range << std::endl;
    if(average<effective_range){

        //have to draw the average of 2d pooints to check
        std::vector<std::vector<Keypoint>> left_2d;
        for (int i=0;i< left.size();i++){


            std::vector<Keypoint> left_converted;

            for (const auto& point : left[i]) {
                // cv::Point center(static_cast<int>(point.x), static_cast<int>(point.y));
                // cv::circle(crop1, center, 2, cv::Scalar(0, 255, 0), -1);
                float x_adj = point.x + std::max(0, static_cast<int>(box1[0] - 0.05 * (box1[2] - box1[0])));
                float y_adj = point.y + std::max(0, static_cast<int>(box1[1] - 0.05 * (box1[3] - box1[1])));


                Keypoint temp={x_adj, y_adj};
                left_converted.push_back(temp);
            }

            left_2d.push_back(left_converted);

        }


        std::vector<Keypoint> averageKeypoints = calculateAverageKeypoints(left_2d);

        for (int i=5; i< averageKeypoints.size();i++){
            cv::Point center(static_cast<int>(averageKeypoints[i].x), static_cast<int>(averageKeypoints[i].y));
            cv::circle(rectifiedL_copy, center, 4, cv::Scalar(0, 255, 0), -1);

        }


        std::size_t lastSlashIndex = imgf1.find_last_of("/\\");
        std::string filename = imgf1.substr(lastSlashIndex + 1); output_folder ;

        std::string outputPath = output_folder  + filename;
        cv::imwrite(outputPath, rectifiedL_copy);


        std::vector<std::vector<float>> variance_vector_list;

        
        for (int i=0;i<vec_inds.size();i++){
            std::vector<float> temp;

            for(int j=0; j<list_of_mag.size(); j++ ){
                if(list_of_mag[j][i] != 0){
                    temp.push_back(list_of_mag[j][i]);
                }
                

            }
            variance_vector_list.push_back(temp);


        }

        int count=0;

        for(int i=0;i<variance_vector_list.size();i++){
            float variance = calculateVariance(variance_vector_list[i]);
            if ((variance>0)&&(variance<variance_threshold )){
                count+=1;
            }
            else{
                variance_vector_list[i]={};
            }
        }

        if (count< required_variance_point){
            return Frame();
        }
        else{

            Frame frame=merge(variance_vector_list, imgf1, imgf2);
            return frame;

        }
    }
    else{
        return Frame();
    }


}




//main function



int main(int argc, char **argv) {



    float ret_val = dist_intersect(1.0f, 1.0f, 6.0f, 1.0f);
    // std::cout << "dist_intersect: " << ret_val << std::endl;

    std::map<float, float> map_probs = gen_dict();

    double distri = 0.677f;

    int boundary_threshold=3;
    float intersect_threshold=2.01e-18;

    std::unique_ptr<tflite::FlatBufferModel> detection_model =
        tflite::FlatBufferModel::BuildFromFile("../detection/yolov8s_integer_quant.tflite");

    //   auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
    //   auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
    tflite::ops::builtin::BuiltinOpResolver detection_resolver;
    std::unique_ptr<tflite::Interpreter> detection_interpreter;
    tflite::InterpreterBuilder(*detection_model, detection_resolver)(&detection_interpreter);
    //   interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);
    detection_interpreter->SetAllowFp16PrecisionForFp32(false);
    detection_interpreter->AllocateTensors();
    //end of detection model


    //for movenet model
    //prepare the intrepreter of movenet    
    std::unique_ptr<tflite::FlatBufferModel> movenet_model=tflite::FlatBufferModel::BuildFromFile("../movenet/movenet.tflite");
    tflite::ops::builtin::BuiltinOpResolver movenet_resolver;
    std::unique_ptr<tflite::Interpreter> movenet_interpreter;
    tflite::InterpreterBuilder(*movenet_model, movenet_resolver)(&movenet_interpreter);

    movenet_interpreter->SetAllowFp16PrecisionForFp32(false);
    movenet_interpreter->AllocateTensors();
    //end of movenet model

    std::string left_dir = "./part/left/";
    std::string right_dir = "./part/right/";
    std::string output_folder = "./output/";

    // Get list of files in left directory
    std::vector<std::string> left_files;
    for (const auto& entry : fs::directory_iterator(left_dir)) {
        if (entry.path().extension() == ".jpg") {
            left_files.push_back(entry.path().filename());
        }
    }
    std::sort(left_files.begin(), left_files.end(), sortNumerically);


    //collect valid frames
    std::vector<std::string> valid_frames_names;

    //collect valid vectors
    std::vector<Frame> valid_frames;


    for (const auto& file_name : left_files) {
        // Replace 'left' with 'right' in the filename
        std::string right_file_name = std::regex_replace(file_name, std::regex("left"), "right");

        std::string left_file_path = left_dir + file_name;
        std::string right_file_path = right_dir + right_file_name;

        std::cout << "-----------"<< left_file_path << right_file_path << std::endl;

        if (fs::exists(right_file_path)) {
            cv::Mat frameR = cv::imread(left_file_path);
            cv::Mat frameL = cv::imread(right_file_path);


            Frame frame=process_eachframe(detection_interpreter, movenet_interpreter, left_file_path, right_file_path, output_folder);
            frame.printMeanAndRange();

            if (!frame.isEmpty()) {
                
                valid_frames_names.push_back(file_name);

                valid_frames.push_back(frame);

            }

        }
    }

    std::cout << "Valid Frames:" << std::endl;
    for (const std::string& frame : valid_frames_names) {
        std::cout << frame << " ";
    }    

    std::vector<std::pair<size_t, size_t>> pair_index;

    if (valid_frames.size() < 1) {
        std::cout << "no enough valid frames" << std::endl;
    } else {
        std::vector<float> adjacent_result;
        for (size_t i = 0; i < valid_frames.size() - 1; ++i) {
            adjacent_result.push_back(intersect(valid_frames[i].mean_local, valid_frames[i + 1].mean_local, valid_frames[i].range_local, valid_frames[i + 1].range_local, distri, map_probs));
        }

        std::cout << "Adjacent Result: ";
        for (const auto& val : adjacent_result) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::vector<size_t> indices_sorted(adjacent_result.size());
        std::iota(indices_sorted.begin(), indices_sorted.end(), 0);
        std::sort(indices_sorted.begin(), indices_sorted.end(), [&](size_t i, size_t j) { return adjacent_result[i] > adjacent_result[j]; });

        
        for (auto i : indices_sorted) {
            pair_index.push_back({i, i + 1});
        }

        std::cout << "Pair Index: ";
        for (const auto& pair : pair_index) {
            std::cout << "(" << pair.first << ", " << pair.second << ") ";
        }
        std::cout << std::endl;
    }

    std::cout << "----------" << std::endl;

    std::vector<Frame> local_database = process_frames(valid_frames, pair_index, intersect_threshold, boundary_threshold, distri, map_probs );

    for (const auto& frame : local_database) {
        frame.printDetails();
        std::cout << "------------------" << std::endl;
    }


    return 0;
}





//to run this: make && ./TFLiteImageClassification


