




#include "movenet.hpp"



//****************************************************
//start to get movenet


//some function for preprocessing for C++ tflite inference
// struct Keypoint {
//     float x;
//     float y;
// };

cv::Mat ResizeWithPad(const cv::Mat& src, int target_width, int target_height) {
    int width = src.cols, height = src.rows;
    cv::Mat dst = src.clone();

    // Calculate the scaling factors for width and height independently
    double scale_width = target_width / static_cast<double>(width);
    double scale_height = target_height / static_cast<double>(height);
    double scale = std::min(scale_width, scale_height);

    // Calculate the new size
    int new_width = static_cast<int>(width * scale);
    int new_height = static_cast<int>(height * scale);

    // Resize the image
    cv::resize(src, dst, cv::Size(new_width, new_height));

    // Calculate padding
    int pad_width = target_width - new_width;
    int pad_height = target_height - new_height;
    int pad_left = pad_width / 2;
    int pad_top = pad_height / 2;
    int pad_right = pad_width - pad_left;
    int pad_bottom = pad_height - pad_top;

    // Pad the resized image
    cv::copyMakeBorder(dst, dst, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return dst;
}




std::vector<Keypoint> process_movenet(const std::unique_ptr<tflite::Interpreter>& interpreter, const cv::Mat& img, float movenet_threshold) {

    int width = img.cols, height = img.rows;
    float scale, x_shift, y_shift;

    if (height > width) {
        y_shift = 0;
        scale = static_cast<float>(height) / 256;
        x_shift = (256 - static_cast<float>(width) / scale) / 2;
    } else {
        x_shift = 0;
        scale = static_cast<float>(width) / 256;
        y_shift = (256 - static_cast<float>(height) / scale) / 2;
    }

    cv::Mat resized_img;

    resized_img = ResizeWithPad(img, 256, 256);

    cv::Mat img_float;

    resized_img.convertTo(img_float, CV_8UC3);


    std::cout << "Got movenet model" << std::endl;

    // Get input & output tensor
    TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);
    TfLiteTensor *output_tensor = interpreter->tensor(interpreter->outputs()[0]);

    //just try the output shape
    // Print the shape of the output tensor
    std::cout << "Output tensor shape: ";
    for (int i = 0; i < output_tensor->dims->size; ++i) {
        std::cout << output_tensor->dims->data[i] << " ";
    }
    std::cout << std::endl;
    //end of output shape

    const uint HEIGHT = input_tensor->dims->data[1];
    const uint WIDTH = input_tensor->dims->data[2];
    // Assuming the input image is already resized to the model's expected size



    // Prepare input tensor
    cv::Mat inputImg = img_float; // Assuming img is already preprocessed
    memcpy(input_tensor->data.f, inputImg.ptr<float>(0), HEIGHT * WIDTH * 3 * sizeof(float));

    // Run inference
    interpreter->Invoke();

    // Retrieve output
    float* output_data = output_tensor->data.f;

    // float *output_data = interpreter->typed_output_tensor<float>(0);

    // Process output
    std::vector<Keypoint> points;
    
    //movenet outputs 17 keypoints pair
    int output_size = 17; // Assuming 17 keypoints
    for (int i = 0; i < output_size; ++i) {
        float y = output_data[i * 3];
        float x = output_data[i * 3 + 1];
        float confidence = output_data[i * 3 + 2];

        if (confidence > movenet_threshold) {
            Keypoint pt = {
                ((-x_shift + (x * 256)) * scale),
                ((-y_shift + (y * 256)) * scale)
            };
            points.push_back(pt);

            // Draw the keypoint on the original image
            cv::circle(img, cv::Point(pt.x, pt.y), 1, cv::Scalar(0, 255, 0), -1);
        } else {
            points.push_back(Keypoint{0, 0});
        }
    }

    // // // Display the image with keypoints
    // cv::imshow("Keypoints", img);
    // cv::waitKey(0); // Wait for a key press to close the window

    for (const auto& point : points) {
        std::cout << "Point: (" << point.x << ", " << point.y << ")" << std::endl;

    }



    return points;

}




// void printKeypoints(const std::vector<std::vector<Keypoint>>& keypoints) {
//     for (size_t i = 0; i < keypoints.size(); ++i) {
//         std::cout << "Vector " << i << ":" << std::endl;
//         for (size_t j = 0; j < keypoints[i].size(); ++j) {
//             std::cout << "Keypoint " << j << ": (" << keypoints[i][j].x << ", " << keypoints[i][j].y << ")" << std::endl;
//         }
//     }
// }







std::vector<std::vector<Keypoint>> process_movenet_augmentation(const std::unique_ptr<tflite::Interpreter>& interpreter, const cv::Mat& img, float movenet_threshold, int loop_threshold, bool use_aug) {


    
    std::vector<std::vector<Keypoint>> outer_list;

    int output_size = 17; // Assuming 17 keypoints


    //default is loops for augmentation
    for(int num=0; num<loop_threshold; num++){

        // Process output
        std::vector<Keypoint> points;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.8, 1.2);
        std::uniform_int_distribution<> dis_int(0, 0);

        // Choose augmentation
        std::string augmentations[3] = {"stretch","noise"};
        std::string augmentation = augmentations[dis_int(gen)];

        cv::Mat M = cv::Mat::eye(3, 3, CV_32F);

        cv::Mat imgr_temp = img.clone();

        int imgr_height = img.rows;
        int imgr_width = img.cols;

        if(use_aug){
          if (augmentation == "noise") {
              cv::Mat noise = cv::Mat(imgr_height, imgr_width, imgr_temp.type());
              cv::randu(noise, cv::Scalar::all(0), cv::Scalar::all(50));
              imgr_temp += noise;
          }

          else if (augmentation == "stretch") {
              float fx = dis(gen), fy = dis(gen);
              cv::resize(imgr_temp, imgr_temp, cv::Size(), fx, fy);
              M.at<float>(0, 0) = fx;
              M.at<float>(1, 1) = fy;
          }       
        }

        // else if (augmentation == "crop") {
        //     int x = rand() % (imgr_width / 4);
        //     int y = rand() % (imgr_height / 4);
        //     int w = imgr_width - 2 * x;
        //     int h = imgr_height - 2 * y;
        //     cv::Rect crop_region(x, y, w, h);
        //     imgr_temp = imgr_temp(crop_region);
        //     M.at<float>(0, 2) = -x;
        //     M.at<float>(1, 2) = -y;
        // }

        

        int width = imgr_temp.cols, height = imgr_temp.rows;
        float scale, x_shift, y_shift;

        if (height > width) {
            y_shift = 0;
            scale = static_cast<float>(height) / 256;
            x_shift = (256 - static_cast<float>(width) / scale) / 2;
        } else {
            x_shift = 0;
            scale = static_cast<float>(width) / 256;
            y_shift = (256 - static_cast<float>(height) / scale) / 2;
        }

        cv::Mat resized_img;
        resized_img = ResizeWithPad(imgr_temp, 256, 256);

        cv::Mat img_input;
        resized_img.convertTo(img_input, CV_8UC3);

        // std::cout << "--------------------" << std::endl;
        //here is correct
        // Get input & output tensor
        TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);
        TfLiteTensor *output_tensor = interpreter->tensor(interpreter->outputs()[0]);

        const uint HEIGHT = input_tensor->dims->data[1];
        const uint WIDTH = input_tensor->dims->data[2];
        cv::Mat inputImg = img_input; // Assuming img is already preprocessed

        // std::cout << "--------------------" << std::endl;

        // memcpy(input_tensor->data.f, inputImg.ptr<float>(0), HEIGHT * WIDTH * 3 * sizeof(float));
        memcpy(input_tensor->data.uint8, inputImg.ptr<uchar>(0), HEIGHT * WIDTH * 3 * sizeof(uchar));

        // std::cout << "--------------------" << std::endl;

        // Run inference
        interpreter->Invoke();

        // Retrieve output
        float* output_data = output_tensor->data.f;
        // float *output_data = interpreter->typed_output_tensor<float>(0);
        
        // cv::imshow("Keypoints", img_input);
        // cv::waitKey(0); // Wait for a key press to close the window

        

        //movenet outputs 17 keypoints pair
        

        for (int i = 0; i < output_size; ++i) {
            float y = output_data[i * 3];
            float x = output_data[i * 3 + 1];
            float confidence = output_data[i * 3 + 2];

            if (confidence > movenet_threshold) {
                Keypoint pt = {
                    ((-x_shift + (x * 256)) * scale),
                    ((-y_shift + (y * 256)) * scale)
                };
                points.push_back(pt);

                // Draw the keypoint on the original image
                // cv::circle(img, cv::Point(pt.x, pt.y), 1, cv::Scalar(0, 255, 0), -1);
            } else {
                points.push_back(Keypoint{0, 0});
            }
        }


        // for (const auto& point : points) {
        //     cv::Point center(static_cast<int>(point.x), static_cast<int>(point.y));
        //     cv::circle(imgr_temp, center, 1, cv::Scalar(0, 255, 0), -1);

        //     // std::cout << "Keypoint (x: " << point.x << ", y: " << point.y << ")\n";
        // }
        // cv::imshow("Keypoints", imgr_temp);
        // cv::waitKey(0); // Wait for a key press to close the window
        

        for (int i = 0; i < points.size(); ++i) {

            auto& point = points[i];  // Get a reference to the point to modify it directly

            if(point.x != 0 && point.y != 0) {
                float x_adj = point.x;
                float y_adj = point.y;

                // Transform points back according to the inverse of M
                if(use_aug){
                  if (augmentation != "noise") {
                      cv::Mat inv_M = M.inv();
                      cv::Vec3f homog_point(x_adj, y_adj, 1);
                      std::vector<cv::Vec3f> transformed_points;
                      cv::transform(std::vector<cv::Vec3f>{homog_point}, transformed_points, inv_M);
                      x_adj = transformed_points[0][0];
                      y_adj = transformed_points[0][1];
                  }
                }

                Keypoint pt = {
                    x_adj,
                    y_adj
                };
                point = pt;  // No need for static_cast

            } 

            else {

                point = Keypoint{0, 0};

            }
        }
        outer_list.push_back(points); 


    }//end of loops


    // //here is right
    // // std::cout << "Movenet DONE, " << outer_list.size() << "found\n";

    // // for (const auto& points : outer_list) {
    // //     for (const auto& point : points) {
    // //         cv::Point center(static_cast<int>(point.x), static_cast<int>(point.y));
    // //         cv::circle(img, center, 1, cv::Scalar(0, 255, 0), -1);

    // //         // std::cout << "111111111111Keypoint (x: " << point.x << ", y: " << point.y << ")\n";
    // //     }
    // // }

    // // cv::imshow("Keypoints", img);
    // // cv::waitKey(0); // Wait for a key press to close the window




    // std::vector<std::vector<float>> xlist, ylist;




    // //outer_list is rows 17 cols
    // for (int i=0;i<output_size;i++){
    //     std::vector<float> xposition, yposition;
    //     for(const auto& keypoints : outer_list){
    //         xposition.push_back(keypoints[i].x);
    //         yposition.push_back(keypoints[i].y);
    //     }
    //     xlist.push_back(xposition);
    //     ylist.push_back(yposition);
    // }



    // // Process each list
    // auto processList = [](std::vector<std::vector<float>>& list) {
    //     for (auto& sublist : list) {
    //         // Remove zeros and sort
    //         sublist.erase(std::remove(sublist.begin(), sublist.end(), 0.0f), sublist.end());
    //         std::sort(sublist.begin(), sublist.end());

    //         // Quartile filtering
    //         size_t n = sublist.size();
    //         size_t lower_index = static_cast<size_t>(std::ceil(n * 0.25));
    //         size_t upper_index = static_cast<size_t>(std::floor(n * 0.75));
    //         sublist = std::vector<float>(sublist.begin() + lower_index, sublist.begin() + upper_index);

    //         // Calculate median
    //         size_t len = sublist.size();
    //         if (len >= 2) {
    //             if (len % 2 == 1) {
    //                 sublist = {sublist[len / 2]};
    //             } else {
    //                 sublist = {(sublist[len / 2 - 1] + sublist[len / 2]) / 2};
    //             }
    //         } else if (len == 1) {
    //             sublist = {sublist[0]};
    //         } else {
    //             sublist = {0};
    //         }
    //     }
    // };

    // // Applying processing to both lists
    // processList(xlist);
    // processList(ylist);

    // std::vector<Keypoint> result;
    // // Pairing up the results
    // assert(xlist.size() == ylist.size());
    // for (size_t i = 0; i < xlist.size(); ++i) {
    //     result.push_back(Keypoint{xlist[i].front(), ylist[i].front()});
    // }



    // return result;



    // printKeypoints(outer_list);




    return outer_list;

}




std::vector<Keypoint> process_movenet_onetime(const std::unique_ptr<tflite::Interpreter>& interpreter, const cv::Mat& img, float movenet_threshold, int loop_threshold) {


    
    std::vector<std::vector<Keypoint>> outer_list;

    int output_size = 17; // Assuming 17 keypoints


    //default is 50 loops
    for(int num=0; num<loop_threshold; num++){

        // Process output
        std::vector<Keypoint> points;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.8, 1.2);
        std::uniform_int_distribution<> dis_int(0, 0);

        // Choose augmentation
        std::string augmentations[3] = {"stretch","noise"};
        std::string augmentation = augmentations[dis_int(gen)];

        cv::Mat M = cv::Mat::eye(3, 3, CV_32F);

        cv::Mat imgr_temp = img.clone();

        int imgr_height = img.rows;
        int imgr_width = img.cols;

        

        int width = imgr_temp.cols, height = imgr_temp.rows;
        float scale, x_shift, y_shift;

        if (height > width) {
            y_shift = 0;
            scale = static_cast<float>(height) / 256;
            x_shift = (256 - static_cast<float>(width) / scale) / 2;
        } else {
            x_shift = 0;
            scale = static_cast<float>(width) / 256;
            y_shift = (256 - static_cast<float>(height) / scale) / 2;
        }

        cv::Mat resized_img;
        resized_img = ResizeWithPad(imgr_temp, 256, 256);

        cv::Mat img_input;
        resized_img.convertTo(img_input, CV_8UC3);

        // std::cout << "--------------------" << std::endl;
        //here is correct
        // Get input & output tensor
        TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);
        TfLiteTensor *output_tensor = interpreter->tensor(interpreter->outputs()[0]);

        const uint HEIGHT = input_tensor->dims->data[1];
        const uint WIDTH = input_tensor->dims->data[2];
        cv::Mat inputImg = img_input; // Assuming img is already preprocessed

        // std::cout << "--------------------" << std::endl;

        // memcpy(input_tensor->data.f, inputImg.ptr<float>(0), HEIGHT * WIDTH * 3 * sizeof(float));
        memcpy(input_tensor->data.uint8, inputImg.ptr<uchar>(0), HEIGHT * WIDTH * 3 * sizeof(uchar));

        // std::cout << "--------------------" << std::endl;

        // Run inference
        interpreter->Invoke();

        // Retrieve output
        float* output_data = output_tensor->data.f;
        // float *output_data = interpreter->typed_output_tensor<float>(0);
        
        // cv::imshow("Keypoints", img_input);
        // cv::waitKey(0); // Wait for a key press to close the window

        

        //movenet outputs 17 keypoints pair
        

        for (int i = 0; i < output_size; ++i) {
            float y = output_data[i * 3];
            float x = output_data[i * 3 + 1];
            float confidence = output_data[i * 3 + 2];

            if (confidence > movenet_threshold) {
                Keypoint pt = {
                    ((-x_shift + (x * 256)) * scale),
                    ((-y_shift + (y * 256)) * scale)
                };
                points.push_back(pt);

                // Draw the keypoint on the original image
                // cv::circle(img, cv::Point(pt.x, pt.y), 1, cv::Scalar(0, 255, 0), -1);
            } else {
                points.push_back(Keypoint{0, 0});
            }
        }


        // for (const auto& point : points) {
        //     cv::Point center(static_cast<int>(point.x), static_cast<int>(point.y));
        //     cv::circle(imgr_temp, center, 1, cv::Scalar(0, 255, 0), -1);

        //     // std::cout << "Keypoint (x: " << point.x << ", y: " << point.y << ")\n";
        // }
        // cv::imshow("Keypoints", imgr_temp);
        // cv::waitKey(0); // Wait for a key press to close the window
        

        for (int i = 0; i < points.size(); ++i) {

            auto& point = points[i];  // Get a reference to the point to modify it directly

            if(point.x != 0 && point.y != 0) {
                float x_adj = point.x;
                float y_adj = point.y;


                Keypoint pt = {
                    x_adj,
                    y_adj
                };
                point = pt;  // No need for static_cast

            } 

            else {

                point = Keypoint{0, 0};

            }
        }
        outer_list.push_back(points); 


    }//end of loops


    //here is right
    // std::cout << "Movenet DONE, " << outer_list.size() << "found\n";

    // for (const auto& points : outer_list) {
    //     for (const auto& point : points) {
    //         cv::Point center(static_cast<int>(point.x), static_cast<int>(point.y));
    //         cv::circle(img, center, 1, cv::Scalar(0, 255, 0), -1);

    //         // std::cout << "111111111111Keypoint (x: " << point.x << ", y: " << point.y << ")\n";
    //     }
    // }

    // cv::imshow("Keypoints", img);
    // cv::waitKey(0); // Wait for a key press to close the window




    std::vector<std::vector<float>> xlist, ylist;




    //outer_list is rows 17 cols
    for (int i=0;i<output_size;i++){
        std::vector<float> xposition, yposition;
        for(const auto& keypoints : outer_list){
            xposition.push_back(keypoints[i].x);
            yposition.push_back(keypoints[i].y);
        }
        xlist.push_back(xposition);
        ylist.push_back(yposition);
    }



    // Process each list
    auto processList = [](std::vector<std::vector<float>>& list) {
        for (auto& sublist : list) {
            // Remove zeros and sort
            sublist.erase(std::remove(sublist.begin(), sublist.end(), 0.0f), sublist.end());
            std::sort(sublist.begin(), sublist.end());

            // Quartile filtering
            size_t n = sublist.size();
            size_t lower_index = static_cast<size_t>(std::ceil(n * 0.25));
            size_t upper_index = static_cast<size_t>(std::floor(n * 0.75));
            sublist = std::vector<float>(sublist.begin() + lower_index, sublist.begin() + upper_index);

            // Calculate median
            size_t len = sublist.size();
            if (len >= 2) {
                if (len % 2 == 1) {
                    sublist = {sublist[len / 2]};
                } else {
                    sublist = {(sublist[len / 2 - 1] + sublist[len / 2]) / 2};
                }
            } else if (len == 1) {
                sublist = {sublist[0]};
            } else {
                sublist = {0};
            }
        }
    };

    // Applying processing to both lists
    processList(xlist);
    processList(ylist);

    std::vector<Keypoint> result;
    // Pairing up the results
    assert(xlist.size() == ylist.size());
    for (size_t i = 0; i < xlist.size(); ++i) {
        result.push_back(Keypoint{xlist[i].front(), ylist[i].front()});
    }



    return result;

}



void drawLinesBetweenPoints(cv::Mat& image, const std::vector<Keypoint>& keypoints, const std::vector<std::vector<int>>& vec_inds) {
    for (const auto& inds : vec_inds) {
        if (inds.size() == 2 && inds[0] < keypoints.size() && inds[1] < keypoints.size()) {
            cv::Point pt1(static_cast<int>(keypoints[inds[0]].x), static_cast<int>(keypoints[inds[0]].y));
            cv::Point pt2(static_cast<int>(keypoints[inds[1]].x), static_cast<int>(keypoints[inds[1]].y));
            cv::line(image, pt1, pt2, cv::Scalar(200, 200, 255), 2); // Lighter red color in BGR format
        }
    }
}



//end of movenet
//**********************************





