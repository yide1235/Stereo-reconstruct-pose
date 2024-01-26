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

#include "opencv481.h"




using namespace std;
using namespace cv;
using namespace tflite;
// ns
using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
// ns
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

typedef cv::Point3_<float> Pixel;



//***************************
//function for intersection function
//already have validated
float phi(float z) {
    return (1.0f + std::erf(z / std::sqrt(2.0f))) / 2.0f;
}

float point_match(float dist1_mu, float dist1_sig, float val) {
    float z = std::abs((val - dist1_mu) / dist1_sig);
    float cum_prob = phi(z);
    return 1.0f - cum_prob;
}

float dist_intersect(float dist1_mu, float dist1_sig, float dist2_mu, float dist2_sig) {
    float step_sig = std::max(dist1_sig, dist2_sig);
    float step = 6.0f * step_sig / 10.0f;
    float startx = std::min(dist1_mu, dist2_mu) - 6.0f * step_sig;
    float endx = std::max(dist1_mu, dist2_mu) + 6.0f * step_sig;
    float int_prob = 0.0f;

    for (float currx = startx; currx < endx; currx += step) {
        float refz1 = (currx - dist1_mu) / dist1_sig;
        float refz2 = ((currx + step) - dist1_mu) / dist1_sig;
        float p1 = phi(refz1);
        float p2 = phi(refz2);
        float prob1 = std::abs(p2 - p1);

        refz1 = (currx - dist2_mu) / dist2_sig;
        refz2 = ((currx + step) - dist2_mu) / dist2_sig;
        p1 = phi(refz1);
        p2 = phi(refz2);
        float prob2 = std::abs(p2 - p1);

        int_prob += std::min(prob1, prob2);
    }

    return int_prob;
}

float est_sig(float rng, float prob, const std::map<float, float>& map_probs) {
    prob = std::round(prob * 100.0f) / 100.0f;
    auto it = map_probs.find(prob);
    if (it != map_probs.end()) {
        return 0.5f * rng / it->second;
    }
    return (prob <= 0.0f || prob > 1.0f) ? -1.0f : -1.0f;
}

std::map<float, float> gen_dict() {
    std::map<float, float> map_probs;
    for (int x = 1; x < 350; ++x) {
        float prob = std::round((phi(static_cast<float>(x) / 100.0f) - phi(static_cast<float>(-x) / 100.0f)) * 100.0f) / 100.0f;
        if (map_probs.find(prob) == map_probs.end()) {
            map_probs[prob] = static_cast<float>(x) / 100.0f;
        }
    }
    return map_probs;
}

std::pair<float, std::vector<float>> intersect(const std::vector<float>& means1, const std::vector<float>& means2, const std::vector<float>& range1, const std::vector<float>& range2, float prob, const std::map<float, float>& map_probs) {
    float mult = 1.0f;
    std::vector<float> probs;
    float tot = 0.0f;
    int cnt = 0, nan_cnt = 0;

    for (size_t i = 0; i < means1.size(); ++i) {
        float sig1 = est_sig(range1[i], prob, map_probs);
        float sig2 = est_sig(range2[i], prob, map_probs);
        if (sig1 == -1.0f || sig2 == -1.0f) {
            probs.push_back(NAN);
            nan_cnt++;
            continue;
        }

        float int_prob = dist_intersect(means1[i], sig1, means2[i], sig2);
        if (int_prob == 0.0f) {
            probs.push_back(NAN);
            nan_cnt++;
            continue;
        }

        mult *= int_prob;
        tot += int_prob;
        cnt++;
        probs.push_back(int_prob);
    }

    float avg_prob = cnt > 0 ? tot / cnt : 0.0f;
    if (nan_cnt > 0) {
        mult *= std::pow(10.0f, -nan_cnt);
    }

    return {mult, probs};
}



// int main() {
//     double ret_val = dist_intersect(1, 1, 6, 1);
//     std::cout << "dist_intersect: " << ret_val << std::endl;

//     std::map<double, double> map_probs = gen_dict();

//     double distri = 0.677;
//     std::vector<double> mean1 = {32.51783905029297, 22.714483642578124, 18.473861694335938, 23.965794372558594, 20.531011962890624, 36.12490539550781, 35.75204772949219, 34.6367431640625, 32.66388854980469, 48.96383972167969, 50.294253540039065, 20.608946228027342};
//     std::vector<double> range1 = {1.5, 0.7120620727539055, 1.5, 0.5262313842773452, 0.833995056152343, 1.004507446289061, 0.8336090087890611, 1.1875183105468778, 1.5, 1.3794799804687514, 1.5, 0.5};

//     std::vector<double> mean2 = {31.859530639648437, 24.62757110595703, 20.4177490234375, 22.398268127441405, 22.508877563476563, 36.436947631835935, 36.81101684570312, 37.32706909179687, 35.57789001464844, 45.0595718383789, 47.786416625976564, 19.50404052734375};
//     std::vector<double> range2 = {0.5, 0.8721343994140653, 1.0946456909179716, 1.4546157836914055, 1.5, 0.5, 0.9813110351562528, 0.9918975830078125, 1.5, 1.5, 0.7932556152343722, 0.7743011474609389};

//     auto [mult, probs] = intersect(mean1, mean2, range1, range2, distri, map_probs);
//     std::cout << "intersect: " << mult << std::endl;
//     for (double prob : probs) {
//         std::cout << prob << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }


//end of intersetction function
//*****************************






//*************************
//function for detection model

auto mat_process(cv::Mat src, uint width, uint height) -> cv::Mat
{
  // convert to float; BGR -> RGB
  cv::Mat dst2;
  // cout << "Creating dst" << endl;
  // // src.convertTo(dst, CV_32FC3);
  // cout << "Creating dst2" << endl;
  cv::cvtColor(src, dst2, cv::COLOR_BGR2RGB);
  // cout << "Creating dst3" << endl;


  cv::Mat normalizedImage(dst2.rows, dst2.cols, CV_32FC3);

  for (int i = 0; i < dst2.rows; i++) {
    for (int j = 0; j < dst2.cols; j++) {
      cv::Vec3b pixel = dst2.at<cv::Vec3b>(i, j);
      cv::Vec3f normalizedPixel;
      // std::cout << static_cast<float>(pixel[0]) / 255.0f << endl;
      normalizedPixel[0] = static_cast<float>(pixel[0]) / 255.0f;
      normalizedPixel[1] = static_cast<float>(pixel[1]) / 255.0f;
      normalizedPixel[2] = static_cast<float>(pixel[2]) / 255.0f;
      normalizedImage.at<cv::Vec3f>(i, j) = normalizedPixel;
    }
  }

 

  return normalizedImage;
}


cv::Mat letterbox(cv::Mat img, int height, int width) {
    cv::Size shape = img.size(); // current shape [height, width]
    cv::Size new_shape(640, 640);
    // cv::Size new_shape(10,10);

    // Scale ratio (new / old)
    float r = std::min(static_cast<float>(new_shape.height) / shape.height,
                        static_cast<float>(new_shape.width) / shape.width);

    // Compute padding
    cv::Size new_unpad(static_cast<int>(std::round(shape.width * r)),
                       static_cast<int>(std::round(shape.height * r)));
    float dw = static_cast<float>(new_shape.width - new_unpad.width);
    float dh = static_cast<float>(new_shape.height - new_unpad.height);

    dw /= 2; // divide padding into 2 sides
    dh /= 2;

    if (shape != new_unpad) { // resize
        cv::resize(img, img, new_unpad, 0, 0, cv::INTER_LINEAR);
    }

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    cv::Mat result_img;
    cv::copyMakeBorder(img, result_img, top, bottom, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    return result_img;
}






//fprintf(stderr, "minimal <external_delegate.so> <tflite model> <use_cache_mode> <cache file> <inputs>\n");
void setupInput(const std::unique_ptr<tflite::Interpreter>& interpreter) {

   auto in_tensor = interpreter->input_tensor(0);

    switch (in_tensor->type) {
      case kTfLiteFloat32:
      {
        std::cout << "datatype for input kTfLiteFloat32" << std::endl;
        //auto in = ReadData(argv[2], input_data, input_idx, in_tensor->bytes);
        //memcpy(interpreter->typed_input_tensor<float>(input_idx), in.data(), in.size());
        break;
      }
      case kTfLiteUInt8:
      {
        std::cout << "datatype for input kTfLiteUInt8" << std::endl;
        //auto in = ReadData(argv[2], input_data, input_idx, in_tensor->bytes);
        //memcpy(interpreter->typed_input_tensor<uint8_t>(input_idx), in.data(), in.size());
        break;
      }
      case kTfLiteInt8: {
        std::cout << "datatype for input kTfLiteInt8" << std::endl;
        //auto in = ReadData(argv[2], input_data, input_idx, in_tensor->bytes);
        //memcpy(interpreter->typed_input_tensor<int8_t>(input_idx), in.data(), in.size());
        break;
      }
      case kTfLiteInt32:
      {
        std::cout << "datatype for input kTfLiteInt32" << std::endl;
        //auto in = ReadData(argv[2], input_data, input_idx, in_tensor->bytes);
        //memcpy(interpreter->typed_input_tensor<int32_t>(input_idx), in.data(), in.size());
        break;
      }
      default: {
        std::cout << "Fatal: datatype for input not implemented" << std::endl;
        //TFLITE_EXAMPLE_CHECK(false);
        break;
      }
    }

    

}

std::vector<float> xywh2xyxy_scale(const std::vector<float>& boxes, float width, float height) {
    std::vector<float> result;
    for (size_t i = 0; i < boxes.size(); i += 4) {
        float x = boxes[i];
        float y = boxes[i + 1];
        float w = boxes[i + 2];
        float h = boxes[i + 3];

        float x1 = (x - w / 2) * width;     // top left x
        float y1 = (y - h / 2) * height;    // top left y
        float x2 = (x + w / 2) * width;     // bottom right x
        float y2 = (y + h / 2) * height;    // bottom right y

        result.push_back(x1);
        result.push_back(y1);
        result.push_back(x2);
        result.push_back(y2);
    }
    return result;
}




std::vector<float> scaleBox(const std::vector<float>& box, int img1Height, int img1Width, int img0Height, int img0Width) {
    std::vector<float> scaledBox = box;

    // Calculate gain and padding
    float gain = std::min(static_cast<float>(img1Height) / img0Height, static_cast<float>(img1Width) / img0Width);
    int padX = static_cast<int>((img1Width - img0Width * gain) / 2 - 0.1);
    int padY = static_cast<int>((img1Height - img0Height * gain) / 2 - 0.1);

    // Apply padding and scaling
    scaledBox[0] -= padX;
    scaledBox[2] -= padX;
    scaledBox[1] -= padY;
    scaledBox[3] -= padY;

    scaledBox[0] /= gain;
    scaledBox[2] /= gain;
    scaledBox[1] /= gain;
    scaledBox[3] /= gain;

    // Clip the box
    scaledBox[0] = std::max(0.0f, std::min(scaledBox[0], static_cast<float>(img0Width)));
    scaledBox[2] = std::max(0.0f, std::min(scaledBox[2], static_cast<float>(img0Width)));
    scaledBox[1] = std::max(0.0f, std::min(scaledBox[1], static_cast<float>(img0Height)));
    scaledBox[3] = std::max(0.0f, std::min(scaledBox[3], static_cast<float>(img0Height)));

    return scaledBox;
}



std::vector<int> NMS(const std::vector<std::vector<float>>& boxes, float overlapThresh) {
    // Return an empty vector if no boxes given
    if (boxes.empty()) {
        return std::vector<int>();
    }

    std::vector<float> x1, y1, x2, y2, areas;
    std::vector<int> indices;



    // Extract coordinates and compute areas

    int a =0;
    while(a<boxes.size()){
        x1.push_back(boxes[a][0]);    
        y1.push_back(boxes[a][1]);
        x2.push_back(boxes[a][2]);
        y2.push_back(boxes[a][3]);
        areas.push_back((x2[a] - x1[a] + 1) * (y2[a] - y1[a] + 1));
        indices.push_back(a);
        a++;
    }


    for(int q=0; q<boxes.size();q++){
      
      std::vector<int>temp_indices;
      for(int p=0;p<indices.size();p++){
        if(indices[p]!=q){
          temp_indices.push_back(indices[p]);
        }
      }
      //q and temp_indices

      vector<float> xx1;
      vector<float> yy1;
      vector<float> xx2;
      vector<float> yy2;


      for(int l=0;l<temp_indices.size();l++){
        xx1.push_back(std::max(boxes[temp_indices[l]][0], boxes[q][0]));
        yy1.push_back(std::max(boxes[temp_indices[l]][1], boxes[q][1]));
        xx2.push_back(std::min(boxes[temp_indices[l]][2], boxes[q][2]));
        yy2.push_back(std::min(boxes[temp_indices[l]][3], boxes[q][3]));
      }

  

      assert( xx2.size() == xx1.size());
      
      vector<float>w;
      for(int x=0; x< xx1.size();x++){

        w.push_back(std::max(0.0f,(xx2[x]-xx1[x]+1)));
        // std::cout << xx2[x] << xx1[x] <<std::endl;

      }

      assert( yy2.size() == yy1.size());

      vector<float>h;
      for(int y=0; y<yy1.size();y++){
          h.push_back(std::max(0.0f, (yy2[y]-yy1[y]+1)));
      }

      vector<float> temp_areas;
      for(int l=0;l<temp_indices.size();l++){
        temp_areas.push_back(areas[temp_indices[l]]);
      }
      // for(int v=0;v<temp_areas.size();v++){
      //     std::cout << temp_areas[v] << std::endl;
      // }

      vector<float> wxh;
      assert(w.size()==h.size());
      for(int b=0;b<w.size();b++){
          wxh.push_back(w[b]*h[b]);
          // std::cout << w[b] << h[b] << std::endl;
          // std::cout << w[b]*h[b] <<std::endl;
      }

      vector<float> overlap;
      assert(wxh.size()==temp_areas.size());
      for(int n=0;n<wxh.size();n++){
          overlap.push_back(wxh[n]/temp_areas[n]);

      }
      bool exist=false;

      for (int u=0;u<overlap.size();u++){
        if(overlap[u]>overlapThresh){
          exist=true;
        }
      }


      if(exist){
        vector<int>temp5;
        for(int w=0;w<indices.size();w++){
          if(indices[w]!=q){
            temp5.push_back(indices[w]);
          }
        }
        indices=temp5;
      }

    }
   

    return indices;
}


//add more area for the bbox when using a tflite model:
// Function to expand bounding boxes
void expandBoundingBoxes(std::vector<std::vector<float>>& boxes, int img_width, int img_height, float width_add, float height_add) {

    // float width_add = 0.03; // 20% increase on each side, making total increase 100% or doubling the size
    // float height_add=0.01;


    for (auto& box : boxes) {

        // std::cout << box[0] << " " << box[1] <<" " <<  box[2] <<" " <<  box[3] <<" " <<  std::endl;

        float x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];

        float width = x2 - x1;
        float height = y2 - y1;

        // Expanding x1 and x2
        float expanded_x1 = std::max(0.0f, x1 - width_add * width);
    
        float expanded_x2 = std::min(static_cast<float>(img_width), x2 + width_add * width);

        // Expanding y1 and y2
        float expanded_y1 = std::max(0.0f, y1 - height_add * height);

        float expanded_y2 = std::min(static_cast<float>(img_height), y2 +  height_add* height);

        // Updating the box
        box[0] = expanded_x1;
        box[1] = expanded_y1;
        box[2] = expanded_x2;
        box[3] = expanded_y2;
    }
}



std::vector<std::vector<float>> process_4(const std::unique_ptr<tflite::Interpreter>& interpreter,const cv::Mat& img, const float detection_threshold)
{


    //setupInput(interpreter);

    // cout << " Got model " << endl;
    // get input & output layer
    TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);
    // cout << " Got input " << endl;
    TfLiteTensor *output_box = interpreter->tensor(interpreter->outputs()[0]);
    // cout << " Got output " << endl;
    // TfLiteTensor *output_score = interpreter->tensor(interpreter->outputs()[1]);
    // cout << " Got output score " << endl;

    const uint HEIGHT = input_tensor->dims->data[1];
    const uint WIDTH = input_tensor->dims->data[2];
    const uint CHANNEL = input_tensor->dims->data[3];
    // cout << "H " << HEIGHT << " W " << WIDTH << " C " << CHANNEL << endl;

    // read image file
    std::chrono::time_point<std::chrono::system_clock> beg, start, end, done, nmsdone;
    std::chrono::duration<double> elapsed_seconds;
    start = std::chrono::system_clock::now();



    const float width=img.rows;
    const float height=img.cols;
    // std::cout << width << height <<std::endl;
    //width is 1080, height is 1920


    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    // printf("Read matrix from file s: %.10f\n", elapsed_seconds.count());

    start = std::chrono::system_clock::now();


    cv::Mat inputImg = letterbox(img, WIDTH, HEIGHT);



    inputImg = mat_process(inputImg, WIDTH, HEIGHT);




    // cout << "DIM IS " << inputImg.channels() << endl;


    // cout << " Got image " << endl;

    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    // printf("Process Matrix to RGB s: %.10f\n", elapsed_seconds.count());
    interpreter->SetAllowFp16PrecisionForFp32(true);

    start = std::chrono::system_clock::now();
    // cout << " GOT INPUT IMAGE " << endl;

    // flatten rgb image to input layer.
    // float* input_data = interpreter->typed_input_tensor<float>(0);
    memcpy(input_tensor->data.f, inputImg.ptr<float>(0), HEIGHT*WIDTH*3* sizeof(float));


    interpreter->Invoke();
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    // printf("invoke interpreter s: %.10f\n", elapsed_seconds.count());

    float *box_vec = interpreter->typed_output_tensor<float>(0);


    int nelem = 1;
    int dim1 = output_box->dims->data[2]; // should be 8400
    int dim2 = output_box->dims->data[1]; // should be 84
    for (int i = 0; i < output_box->dims->size; ++i)
    {
    // cout << "DIM IS " << output_box->dims->data[i] << endl;
    nelem *= output_box->dims->data[i];
    }

    //implemented a faster way in C++

    //use this
    std::vector<float> confidence;
    std::vector<float> index;
    std::vector<std::vector<float>> bbox;



    int base = 4 * 8400;
    int m = 0;


    while (m < 8400) {
    std::vector<float> temp;
    int n = 0;


    while (n < 80) {
        if (box_vec[n * 8400 + m + base] >= detection_threshold) {
        index.push_back(n);
        confidence.push_back(box_vec[n * 8400 + m + base]);


        std::vector<float> temp2;
        int i = 0;
        while (i < 4) {
            temp2.push_back(box_vec[i * 8400 + m]);
            i++;
        }


        // bbox.push_back(xywh2xyxy_scale(temp2,640,640));
        bbox.push_back(temp2);
        }
        n++;
    }
    m++;
    }



    for(int q=0;q<bbox.size();q++){
        bbox[q]=xywh2xyxy_scale(bbox[q],640,640);

    }

    std::vector<std::vector<int>> ind;
    for(int i=0;i< 80;i++){
    std::vector<int> temp4;
    for(int j=0;j<index.size();j++){
        if(i==index[j]){
        // std::cout <<index[j] <<std::endl;
        temp4.push_back(j);
        }
    }

    if(temp4.empty()){
        temp4.push_back(8401);
    }
    ind.push_back(temp4);
    // std::cout << temp4.size() << std::endl;

    }



    //here i have ind, confidence, index, bbox
    std::vector<std::vector<float>> results;

    int confidence_length=confidence.size();
    int class_length=index.size();
    assert(confidence_length == class_length);

    assert(confidence_length == bbox.size());

    std::vector<std::vector<float>> temp_results;

    for (int i=0; i< 80; i++){

        std::vector<std::vector<float>>box_selected;


        std::vector<std::vector<float>> box_afternms;
        // std::vector<float> confidence_selected;
        
        if (ind[i][0]!=8401){
        for(int j=0;j<ind[i].size();j++){
            box_selected.push_back(bbox[ind[i][j]]);
            
        }



        std::vector<int> indices=NMS(box_selected, 0.45);
        if(indices.size()>0){
            for(int s=0;s<indices.size();s++){
            box_afternms.push_back(bbox[ind[i][indices[s]]]);
            }
        }

        for(int d=0;d<box_afternms.size();d++){
            box_afternms[d]=scaleBox(box_afternms[d], HEIGHT, WIDTH, static_cast<int>(width), static_cast<int>(height) );
        }

        vector<float> confidence_afternms;
        if(indices.size()>0){
            for(int s=0;s<indices.size();s++){
            confidence_afternms.push_back(confidence[ind[i][indices[s]]]);
            }
        }

        assert(box_afternms.size()==confidence_afternms.size());

        
        // std::cout << box_afternms.size() << std::endl;
        for(int f=0;f<box_afternms.size();f++){
            vector<float> temp6;
            temp6.push_back(box_afternms[f][0]);
            temp6.push_back(box_afternms[f][1]);
            temp6.push_back(box_afternms[f][2]);
            temp6.push_back(box_afternms[f][3]);
            temp6.push_back(confidence_afternms[f]);
            temp6.push_back(static_cast<float>(i));
            temp_results.push_back(temp6);
        }

        }
        
    }//end of forloop


    //aobe is totally correct

    //img is the image
    int size_threshold=3872;
    // std::cout << temp_results.size() << std::endl;
    //compare results in temp_results with threshold
    for(int i=0; i<temp_results.size();i++){
    std::vector<float> temp7;
    std::vector<int> box7;
    box7.push_back(std::round(temp_results[i][0]));
    box7.push_back(std::round(temp_results[i][1]));
    box7.push_back(std::round(temp_results[i][2]));
    box7.push_back(std::round(temp_results[i][3]));
    cv::Mat detected=img(cv::Rect(box7[0], box7[1], box7[2] - box7[0], box7[3] - box7[1]));
    // std::cout << detected.rows*detected.cols << std::endl;
    if(detected.rows*detected.cols >= size_threshold){
        temp7.push_back(temp_results[i][0]);
        temp7.push_back(temp_results[i][1]);
        temp7.push_back(temp_results[i][2]);
        temp7.push_back(temp_results[i][3]);
        temp7.push_back(temp_results[i][4]);
        temp7.push_back(temp_results[i][5]);
        results.push_back(temp7);
    }

    }

    // for (const std::vector<float>& row :results) {
    //   // Iterate through the elements in each row (inner vector)
    //   for (float element : row) {
    //       std::cout << element << ' ';
    //   }
    //   std::cout << std::endl; // Print a newline after each row
    // }

    // the last two digits is percentage of add to width and height


    expandBoundingBoxes(results,img.cols,img.rows, 0.04, 0.02);

    return results;

}


cv::Scalar hex2rgb(const std::string& h) {
    return cv::Scalar(std::stoi(h.substr(1, 2), 0, 16), std::stoi(h.substr(3, 2), 0, 16), std::stoi(h.substr(5, 2), 0, 16));
}

cv::Scalar getColor(int i, bool bgr = false) {
    std::string hex[] = {
        "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
        "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7"
    };
    int n = sizeof(hex) / sizeof(hex[0]);
    cv::Scalar c = hex2rgb('#' + hex[i % n]);
    return bgr ? cv::Scalar(c[2], c[1], c[0]) : c;
}

cv::Mat plotOneBox(const std::vector<float>& x, cv::Mat im, cv::Scalar color = cv::Scalar(128, 128, 128),
                   const std::string& label = "", int rectLineThickness = 3, int textLineThickness = 2) {
    cv::Point c1(static_cast<int>(x[0]), static_cast<int>(x[1]));
    cv::Point c2(static_cast<int>(x[2]), static_cast<int>(x[3]));
    
    // Draw the rectangle with the specified line thickness
    cv::rectangle(im, c1, c2, color, rectLineThickness, cv::LINE_AA);
    
    if (!label.empty()) {
        int tf = std::max(textLineThickness, 1); // Font thickness
        
        cv::Size textSize = cv::getTextSize(label, 0, 1, tf, nullptr);
        c2 = cv::Point(c1.x + textSize.width, c1.y - textSize.height - 3);
        
        // Draw the filled rectangle for the text background
        cv::rectangle(im, c1, c2, color, -1, cv::LINE_AA);
        
        // Draw the text with the specified line thickness
        cv::putText(im, label, cv::Point(c1.x, c1.y - 2), 0, 1, cv::Scalar(255, 255, 255), tf, cv::LINE_AA);
    }
    
    return im;
}

void plotBboxes(const cv::Mat& img, const std::vector<std::vector<float>>& results,
                const std::vector<std::string>& coco_names, const std::string& savePath) {
    // cv::Mat im0 = cv::imread(imgPath);
    cv::Mat im0=img;
    for (int i = 0; i < results.size(); ++i) {
        const std::vector<float>& value = results[i];
        std::vector<float> bbox(value.begin(), value.begin() + 4);
        float confidence = value[4];
        int clsId = static_cast<int>(value[5]);
        std::string clsName = coco_names[clsId];

  
        // std::cout << trackingid << std::endl;
        // Include tracking ID, class name, and confidence in the label
        
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << confidence;
        std::string formattedValue = ss.str();
        std::string label = clsName  + " " + formattedValue;
        

        cv::Scalar color = getColor(clsId, true);

        im0 = plotOneBox(bbox, im0, color, label);
    }

    try {
        cv::imwrite(savePath, im0);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

//end of detection model
//****************************************************

















//****************************************************
//start to get movenet


//some function for preprocessing for C++ tflite inference
struct Keypoint {
    float x;
    float y;
};

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







std::vector<std::vector<Keypoint>> process_movenet_augmentation(const std::unique_ptr<tflite::Interpreter>& interpreter, const cv::Mat& img, float movenet_threshold, int loop_threshold, bool use_aug=true) {


    
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








//**********************************
//start function for triangulation
bool load_stereo_coefficients(const std::string &filename, cv::Mat &K1, cv::Mat &D1, cv::Mat &K2, cv::Mat &D2, cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F, cv::Size &imageSize) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Could not open stereo calibration file " << filename << std::endl;
        return false;
    }

    fs["K1"] >> K1;
    K1.convertTo(K1, CV_64F); // Ensure matrix is of type double
    fs["D1"] >> D1;
    D1.convertTo(D1, CV_64F); // Ensure matrix is of type double
    fs["K2"] >> K2;
    K2.convertTo(K2, CV_64F); // Ensure matrix is of type double
    fs["D2"] >> D2;
    D2.convertTo(D2, CV_64F); // Ensure matrix is of type double
    fs["R"] >> R;
    R.convertTo(R, CV_64F); // Ensure matrix is of type double
    fs["T"] >> T;
    T.convertTo(T, CV_64F); // Ensure matrix is of type double
    fs["E"] >> E;
    E.convertTo(E, CV_64F); // Ensure matrix is of type double
    fs["F"] >> F;
    F.convertTo(F, CV_64F); // Ensure matrix is of type double

    cv::Mat sizeMat;
    fs["size"] >> sizeMat;
    imageSize = cv::Size(static_cast<int>(sizeMat.at<double>(0)), static_cast<int>(sizeMat.at<double>(1)));

    fs.release();
    return true;
}


std::map<std::string, cv::Mat> get_stereo_coefficients(const std::string &stereo_file, bool rectify = true) {
    cv::Mat K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q;
    cv::Size size;

    if (!load_stereo_coefficients(stereo_file, K1, D1, K2, D2, R, T, E, F, size)) {
        std::cerr << "Error loading stereo coefficients from file" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::map<std::string, cv::Mat> config;
    config["K1"] = K1;
    config["D1"] = D1;
    config["K2"] = K2;
    config["D2"] = D2;
    config["R"] = R;
    config["T"] = T;
    config["E"] = E;
    config["F"] = F;
    config["size"] = cv::Mat(1, 2, CV_32SC1, cv::Scalar(size.width, size.height));





    if (rectify) {
        cv::Mat left_map_x, left_map_y, right_map_x, right_map_y;


        //***********have to use opencv4.8.1 for now, opencv4.5 gives different results!!!!!!!!!!
        // cv::stereoRectify(K1, D1, K2, D2, size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0.9);
        stereoRectify(K1, D1, K2, D2, size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0.9);

        // for (int i = 0; i < R1.rows; ++i) {
        //     for (int j = 0; j < R1.cols; ++j) {
        //         // Adjust the type based on the actual data type of dispL
        //         std::cout <<R1.at<double>(i, j) << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // for (int i = 0; i < P1.rows; ++i) {
        //     for (int j = 0; j < P1.cols; ++j) {
        //         // Adjust the type based on the actual data type of dispL
        //         std::cout <<P1.at<double>(i, j) << " ";
        //     }
        //     std::cout << std::endl;
        // }

        //***********have to use opencv4.8.1 for now, opencv4.5 gives different results!!!!!!!!!!
        cv::initUndistortRectifyMap(K1, D1, R1, P1, size, CV_32FC1, left_map_x, left_map_y);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, size, CV_32FC1, right_map_x, right_map_y);

        // std::cout << left_map_x.at<float>(499, 500) << " ";

        config["R1"] = R1;
        config["R2"] = R2;
        config["P1"] = P1;
        config["P2"] = P2;
        config["Q"] = Q;
        config["left_map_x"] = left_map_x;
        config["left_map_y"] = left_map_y;
        config["right_map_x"] = right_map_x;
        config["right_map_y"] = right_map_y;
    }
    // Q.at<double>(2, 3)=922.8335512009055; //opencv 4.5 and opencv 4.6 gives different valuer for focal length
    double focal_length = Q.at<double>(2, 3);
    
    double baseline = 1.0 / Q.at<double>(3, 2);
    // std::cout << "Q=\n" << Q << "\nfocal_length=" << focal_length << std::endl;
    // std::cout << "T=\n" << T << "\nbaseline    =" << baseline << "mm" << std::endl;

    return config;
}


cv::Mat get_3dpoints(const cv::Mat& disparity, const cv::Mat& Q, float scale = 1.0f) {
    cv::Mat points3D;

    //***********have to use opencv4.8.1 for now, opencv4.5 gives different results!!!!!!!!!!
    cv::reprojectImageTo3D(disparity, points3D, Q);

    // Scale the 3D points
    for (int i = 0; i < points3D.rows; ++i) {
        for (int j = 0; j < points3D.cols; ++j) {
            cv::Point3f& pt = points3D.at<cv::Point3f>(i, j);
            pt.x *= scale;
            pt.y *= scale;
            pt.z *= scale;
        }
    }

    return points3D;
}




// Function to rectify left and right images
std::pair<cv::Mat, cv::Mat> get_rectify_image(const cv::Mat &imgL, const cv::Mat &imgR, const std::map<std::string, cv::Mat> &camera_config) {
    cv::Mat left_map_x = camera_config.at("left_map_x");
    cv::Mat left_map_y = camera_config.at("left_map_y");
    cv::Mat right_map_x = camera_config.at("right_map_x");
    cv::Mat right_map_y = camera_config.at("right_map_y");

    cv::Mat rectifiedL, rectifiedR;


    //***********have to use opencv4.8.1 for now, opencv4.5 gives different results!!!!!!!!!!
    cv::remap(imgL, rectifiedL, left_map_x, left_map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(imgR, rectifiedR, right_map_x, right_map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);




    return {rectifiedL, rectifiedR};
}


//calcuate 3d and depth:
// Function to create a StereoSGBM object

//***********have to use opencv4.8.1 for now, opencv4.5 gives different results!!!!!!!!!!
cv::Ptr<cv::StereoSGBM> StereoSGBM_create(int minDisparity, int numDisparities, int blockSize, int P1, int P2, 
                                          int disp12MaxDiff, int preFilterCap, int uniquenessRatio, 
                                          int speckleWindowSize, int speckleRange, int mode) {
    return cv::StereoSGBM::create(minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff,
                                  preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, mode);
}

// Function to reproject a disparity image to 3D

cv::Mat reprojectImageTo3D(const cv::Mat& disparity, const cv::Mat& Q, cv::Mat* _3dImage = nullptr, 
                           bool handleMissingValues = false, int ddepth = -1) {
    cv::Mat points3D;
    //***********have to use opencv4.8.1 for now, opencv4.5 gives different results!!!!!!!!!!
    cv::reprojectImageTo3D(disparity, points3D, Q, handleMissingValues, ddepth);
    if (_3dImage) {
        *_3dImage = points3D;
    }
    return points3D;
}

// Function to get depth from a disparity image
cv::Mat get_depth(const cv::Mat& disparity, const cv::Mat& Q, float scale = 1.0, bool method = true) {
    cv::Mat depth;
    if (method) {
        cv::Mat points3D = reprojectImageTo3D(disparity, Q);
        std::vector<cv::Mat> channels(3);
        cv::split(points3D, channels);
        depth = channels[2]; // Z channel
    } else {
        float baseline = 1.0f / static_cast<float>(Q.at<double>(3, 2));
        float fx = std::abs(static_cast<float>(Q.at<double>(2, 3)));
        depth = (fx * baseline) / disparity;
    }
    depth *= scale;
    return depth;
}

// WLSFilter class
class WLSFilter {
public:
    WLSFilter(cv::Ptr<cv::StereoMatcher> left_matcher, double lmbda = 80000, double sigma = 1.3) {
        //***********have to use opencv4.8.1 for now, opencv4.5 gives different results!!!!!!!!!!
        filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
        filter->setLambda(lmbda);
        filter->setSigmaColor(sigma);
    }

    void disparity_filter(cv::Mat& dispL, const cv::Mat& imgL, const cv::Mat& dispR) {
        filter->filter(dispL, imgL, cv::noArray(), dispR);
    }

private:
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> filter;
};

// Function to get filtered disparity
cv::Mat get_filter_disparity( cv::Mat& imgL,  cv::Mat& imgR, bool use_wls = true) {
    

    // if (imgL.cols > 500 && imgL.rows > 500) {
    //     uchar pixelValue = imgL.at<uchar>(500, 500);
    //     uchar pixelValue2 = imgL.at<uchar>(500, 500);
    //     std::cout << "Pixel value at (500, 500): " << static_cast<int>(pixelValue) <<static_cast<int>(pixelValue2) << std::endl;
    // } else {
    //     std::cout << "Pixel coordinates are out of bounds." << std::endl;
    // }



    int channels = imgL.channels();
    int blockSize = 3;


    //check this link to know the order of parameter!!!!!!!!: https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
    cv::Ptr<cv::StereoSGBM> matcherL = cv::StereoSGBM::create(
        0,                  // minDisparity
        5 * 16,             // numDisparities
        blockSize,          // blockSize
        8 * 3 * blockSize,  // P1
        32 * 3 * blockSize, // P2
        12,                 // disp12MaxDiff
        63,                 // preFilterCap
        10,                 // uniquenessRatio
        50,                 // speckleWindowSize
        32,                 // speckleRange
        cv::StereoSGBM::MODE_SGBM_3WAY
    );

    // std::cout << "minDisparity: " << matcherL->getMinDisparity() << std::endl;
    // std::cout << "numDisparities: " << matcherL->getNumDisparities() << std::endl;
    // std::cout << "blockSize: " << matcherL->getBlockSize() << std::endl;
    // std::cout << "P1: " << matcherL->getP1() << std::endl;
    // std::cout << "P2: " << matcherL->getP2() << std::endl;
    // std::cout << "disp12MaxDiff: " << matcherL->getDisp12MaxDiff() << std::endl;
    // std::cout << "uniquenessRatio: " << matcherL->getUniquenessRatio() << std::endl;
    // std::cout << "speckleWindowSize: " << matcherL->getSpeckleWindowSize() << std::endl;
    // std::cout << "speckleRange: " << matcherL->getSpeckleRange() << std::endl;
    // std::cout << "preFilterCap: " << matcherL->getPreFilterCap() << std::endl;
    // std::cout << "mode: " << matcherL->getMode() << std::endl;

    //here imgL and imgR is uchar
    //coverts to uint in python number
    // Convert imgL to CV_8U data type
    
    if (imgL.depth() != CV_8U) {
        cv::Mat tempL;
        imgL.convertTo(tempL, CV_8U);
        imgL = tempL;
    }

    // Convert imgR to CV_8U data type
    if (imgR.depth() != CV_8U) {
        cv::Mat tempR;
        imgR.convertTo(tempR, CV_8U);
        imgR = tempR;
    }



    // // Check if the coordinates are within the bounds of imgL
    // if (500 < imgL.rows && 500 < imgL.cols) {
    //     // Get the value at (500,500) in imgL
    //     uchar valueL = imgL.at<uchar>(500, 500);
    //     std::cout << "Value at (500,500) in imgL: " << static_cast<int>(valueL) << std::endl;
    // } else {
    //     std::cout << "Coordinates (500,500) are out of bounds in imgL." << std::endl;
    // }

    // // Check if the coordinates are within the bounds of imgR
    // if (500 < imgR.rows && 500 < imgR.cols) {
    //     // Get the value at (500,500) in imgR
    //     uchar valueR = imgR.at<uchar>(500, 500);
    //     std::cout << "Value at (500,500) in imgR: " << static_cast<int>(valueR) << std::endl;
    // } else {
    //     std::cout << "Coordinates (500,500) are out of bounds in imgR." << std::endl;
    // }

    // //from here imgL and imgR are the same


    cv::Mat dispL;
    //***********have to use opencv4.8.1 for now, opencv4.5 gives different results!!!!!!!!!!
    matcherL->compute(imgL, imgR, dispL); // Corrected compute call

    // dispL.convertTo(dispL, CV_32F, 1.0/16);


    //*********************************
    //this is test code for this
    // // Ensure the requested coordinates are within the matrix bounds
    // if (500 < dispL.rows && 500 < dispL.cols) {
    //     // Assuming dispL is of type CV_16S (signed 16-bit integer)
    //     if (dispL.type() == CV_16S) {
    //         short value = dispL.at<short>(500, 500);
    //         short value1 = dispL.at<short>(500, 501);
    //         short value2 = dispL.at<short>(500, 502);
    //         short value3 = dispL.at<short>(500, 503);
    //         short value4 = dispL.at<short>(500, 504);
    //         std::cout << "Value at (500,501,502,503,504): " << value << value1<< value2<< value3<< value4<< std::endl;
    //     }
    //     // If dispL is of another type, you'll need to adjust the code accordingly.
    //     // For example, if dispL is CV_8U (unsigned 8-bit integer), use:
    //     // unsigned char value = dispL.at<unsigned char>(500, 500);
    // }
    // else {
    //     std::cout << "Coordinates (500,500) are out of bounds." << std::endl;
    // }




    dispL.convertTo(dispL, CV_16S);
    





    if (use_wls) {
        // Create the right matcher and compute the right disparity map
        //***********have to use opencv4.8.1 for now, opencv4.5 gives different results!!!!!!!!!!
        cv::Ptr<cv::StereoMatcher> matcherR = cv::ximgproc::createRightMatcher(matcherL);
        cv::Mat dispR;
        matcherR->compute(imgR, imgL, dispR); // Corrected compute call
        dispR.convertTo(dispR, CV_16S);

        // Create and configure the WLS filter
        double lambda = 80000.0;
        double sigma = 1.3;
        //***********have to use opencv4.8.1 for now, opencv4.5 gives different results!!!!!!!!!!
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wlsFilter = cv::ximgproc::createDisparityWLSFilter(matcherL);
        wlsFilter->setLambda(lambda);
        wlsFilter->setSigmaColor(sigma);

        // Apply the WLS filter
        cv::Mat filteredDispL;
        wlsFilter->filter(dispL, imgL, filteredDispL, dispR);
        filteredDispL.convertTo(filteredDispL, CV_16S);

        dispL = filteredDispL;
    }

    // // Ensure the requested coordinates are within the matrix bounds
    // if (500 < dispL.rows && 500 < dispL.cols) {
    //     // Assuming dispL is of type CV_16S (signed 16-bit integer)
    //     if (dispL.type() == CV_16S) {
    //         short value = dispL.at<short>(500, 500);
    //         std::cout << "Value at (500,500): " << value << std::endl;
    //     }
    //     // If dispL is of another type, you'll need to adjust the code accordingly.
    //     // For example, if dispL is CV_8U (unsigned 8-bit integer), use:
    //     // unsigned char value = dispL.at<unsigned char>(500, 500);
    // }
    // else {
    //     std::cout << "Coordinates (500,500) are out of bounds." << std::endl;
    // }



    
    dispL = cv::max(dispL, 0);
    dispL.convertTo(dispL, CV_32F, 1.0 / 16);

    // // Ensure the requested coordinates are within the matrix bounds
    // if (500 < dispL.rows && 500 < dispL.cols) {
    //     // Assuming dispL is of type CV_16S (signed 16-bit integer)
    //     if (dispL.type() == CV_32F) {
    //         float value = dispL.at<float>(500, 500);
    //         std::cout << "Value at (500,500): " << value << std::endl;
    //     }
    //     // If dispL is of another type, you'll need to adjust the code accordingly.
    //     // For example, if dispL is CV_8U (unsigned 8-bit integer), use:
    //     // unsigned char value = dispL.at<unsigned char>(500, 500);
    // }
    // else {
    //     std::cout << "Coordinates (500,500) are out of bounds." << std::endl;
    // }




    return dispL;//the outputs format is CV_32F

}





//end of triangulation

void printKeypoints(const std::vector<Keypoint>& keypoints) {
    for (const auto& kp : keypoints) {
        std::cout << "Keypoint: (x: " << kp.x << ", y: " << kp.y << ")" << std::endl;
        // Print other properties if any
    }
}



void printDepthValues(const cv::Mat& depth, int numRowsToPrint = 5, int numColsToPrint = 5) {
    std::cout << "Depth values:" << std::endl;
    for (int i = 0; i < std::min(depth.rows, numRowsToPrint); ++i) {
        for (int j = 0; j < std::min(depth.cols, numColsToPrint); ++j) {
            std::cout << depth.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}


std::vector<float> calculateDistances(
    const std::vector<cv::Vec3f>& depth_3d,
    const std::vector<std::vector<int>>& vec_inds) {

    std::vector<float> distances;

    for (const auto& pair : vec_inds) {
        if (depth_3d[pair[0]] == cv::Vec3f(0, 0, 0) || depth_3d[pair[1]] == cv::Vec3f(0, 0, 0)) {
            // If either of the points is (0, 0, 0)
            distances.push_back(0);
        } else {
            // Calculate Euclidean distance
            float dx = depth_3d[pair[0]][0] - depth_3d[pair[1]][0];
            float dy = depth_3d[pair[0]][1] - depth_3d[pair[1]][1];
            float dz = depth_3d[pair[0]][2] - depth_3d[pair[1]][2];
            float distance = std::sqrt(dx * dx + dy * dy + dz * dz) / 10.0f;
            distances.push_back(distance);
        }
    }

    return distances;
}


// Function to print the 3D points
void printDepth3D(const std::vector<cv::Vec3f>& depth_3d) {
    for (const auto& point : depth_3d) {
        std::cout << "Point (x, y, depth): (" << point[0] << ", " << point[1] << ", " << point[2] << ")" << std::endl;
    }
}

//end of triangulation functions
//****************************************



//start of post processing mag algorithm
//****************************************
//********havent validated all
void printNestedVector(const std::vector<std::vector<float>>& nestedVector) {
    for (size_t i = 0; i < nestedVector.size(); ++i) {
        std::cout << "Inner vector " << i << ": ";
        for (size_t j = 0; j < nestedVector[i].size(); ++j) {
            std::cout << nestedVector[i][j] << " ";
        }
        std::cout << std::endl; // Print a newline at the end of each inner vector
    }
}


float calculateVariance(const std::vector<float>& data) {
    if (data.empty()) {
        return 0.0f; // Handle empty vector case
    }

    // Step 1: Calculate the mean of non-zero values
    float sum = 0.0f;
    size_t count = 0;
    for (float value : data) {
        if (value != 0.0f) {
            sum += value;
            ++count;
        }
    }
    if (count == 0) {
        return 0.0f; // All values are zero or empty data
    }
    float mean = sum / count;

    // Step 2: Calculate the sum of squared differences from the mean for non-zero values
    float varianceSum = 0.0f;
    for (float value : data) {
        if (value != 0.0f) {
            varianceSum += (value - mean) * (value - mean);
        }
    }

    // Step 3: Calculate the variance
    float variance = varianceSum / count;
    return variance;
}



class Frame {
public:
    std::vector<float> mean_local;
    std::vector<float> range_local;
    std::vector<std::vector<float>> parts_local;
    std::string frame_name;

    // Default constructor
    Frame() = default;

    // Existing constructor
    Frame(const std::vector<float>& mean_local, const std::vector<float>& range_local, 
          const std::vector<std::vector<float>>& parts_local, const std::string& frame_name)
        : mean_local(mean_local), range_local(range_local), 
          parts_local(parts_local), frame_name(frame_name) {}

    void printMeanAndRange() const {
        std::cout << "mean: ";
        for (float value : mean_local) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        std::cout << "range: ";
        for (float value : range_local) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    bool isEmpty() const {
        return mean_local.empty() && range_local.empty() && parts_local.empty() && frame_name.empty();
    }

    void printDetails() const {
        std::cout << "Frame Name: " << frame_name << std::endl;
        printMeanAndRange();
        // std::cout << "Parts Local:" << std::endl;
        // for (const auto& part : parts_local) {
        //     for (float value : part) {
        //         std::cout << value << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }


};


Frame merge(const std::vector<std::vector<float>>& merge_parts, const std::string& string_a, const std::string& string_b) {
    std::vector<float> ranges;
    std::vector<float> means;
    std::vector<std::vector<float>> result_part;

    // std::cout << "-------------" << std::endl;
    // printNestedVector(merge_parts);
    
    for (const auto& part : merge_parts) {
        if (part.empty()) {
            // continue; // Skip empty parts
            result_part.push_back({});
            ranges.push_back(0);
            means.push_back(0);

        }

        std::vector<float> temp = part;
        std::sort(temp.begin(), temp.end());
        size_t n = temp.size();

        float range1 = 0.5f;
        float mean1 = 0.0f;
        size_t low_bound = static_cast<size_t>(0.4 * n);
        size_t up_bound = static_cast<size_t>(0.6 * n);

        if (n > 5) {
            range1 = temp[up_bound] - temp[low_bound];
            mean1 = temp[n / 2]; // Median for odd number of elements
            if (n % 2 == 0) {
                mean1 = (temp[n / 2 - 1] + temp[n / 2]) / 2.0f; // Median for even number of elements
            }
        } else if (n == 5) {
            range1 = (temp[3] - temp[1]) / 2;
            mean1 = temp[2];
        } else if (n == 4) {
            range1 = (temp[2] - temp[1]) / 2;
            mean1 = (temp[1] + temp[2]) / 2;
        } else if (n == 3) {
            mean1 = temp[1];
        } else if (n == 2) {
            mean1 = (temp[0] + temp[1]) / 2;
        } else if ((n == 1)&&(temp[0]!=0)) {
            mean1 = temp[0];
        }

        range1 = std::max(0.5f, std::min(range1, 1.5f));

        //assume no people segments could be ver 1 meters
        // assert ( mean1<100);
        if(mean1>100){
            temp={};
            range1=0;
            mean1=0;
        }

        result_part.push_back(temp);
        ranges.push_back(range1);
        means.push_back(mean1);
    }

    std::string merged_file_name = string_a + string_b;
    return Frame(means, ranges, result_part, merged_file_name);
}


// Function to extract numerical part from filename and sort based on it
bool sortNumerically(const std::string& a, const std::string& b) {
    std::regex rgx("([0-9]+)");
    std::smatch matchA, matchB;

    if (std::regex_search(a, matchA, rgx) && std::regex_search(b, matchB, rgx)) {
        int numA = std::stoi(matchA[1]);
        int numB = std::stoi(matchB[1]);
        return numA < numB;
    }
    return a < b; // Fallback to lexicographical sorting
}


void printKeypoints(const std::vector<std::vector<Keypoint>>& keypoints) {
    for (size_t i = 0; i < keypoints.size(); ++i) {
        std::cout << "Vector " << i << ":" << std::endl;
        for (size_t j = 0; j < keypoints[i].size(); ++j) {
            std::cout << "Keypoint " << j << ": (" << keypoints[i][j].x << ", " << keypoints[i][j].y << ")" << std::endl;
        }
    }
}


std::vector<Keypoint> calculateAverageKeypoints(const std::vector<std::vector<Keypoint>>& keypoints) {
    if (keypoints.empty()) {
        return {}; // Return empty vector if input is empty
    }

    size_t numKeypoints = keypoints[0].size(); // Number of keypoints in each vector
    std::vector<Keypoint> averages(numKeypoints, {0, 0}); // Initialize averages with zero
    std::vector<int> validCounts(numKeypoints, 0); // Count of valid (non-zero) keypoints

    for (const auto& vec : keypoints) {
        if (vec.size() != numKeypoints) {
            std::cerr << "Inconsistent number of keypoints in vectors" << std::endl;
            return {}; // Handle error or inconsistency
        }
        for (size_t i = 0; i < numKeypoints; ++i) {
            if (vec[i].x != 0 && vec[i].y != 0) {
                averages[i].x += vec[i].x;
                averages[i].y += vec[i].y;
                validCounts[i]++;
            }
        }
    }

    for (size_t i = 0; i < numKeypoints; ++i) {
        if (validCounts[i] > 0) {
            averages[i].x /= validCounts[i];
            averages[i].y /= validCounts[i];
        }
    }

    return averages;
}


std::vector<float> findLargestBBox(const std::vector<std::vector<float>>& bboxes) {
    std::vector<float> largestBBox;
    float maxArea = 0.0f;

    for (const auto& bbox : bboxes) {
        // Ensure the class is 0
        if (bbox[5] == 0) {
            float width = bbox[2] - bbox[0];
            float height = bbox[3] - bbox[1];
            float area = width * height;

            if (area > maxArea) {
                maxArea = area;
                largestBBox = bbox;
            }
        }
    }

    return largestBBox;
}




void printListOfFrames(const std::vector<Frame>& frames) {
    for (const auto& frame : frames) {
        frame.printDetails();
        std::cout << "---------------------" << std::endl;
    }
}





//several function are not validated



//****************************************
//end of post processing mag algorithm





//main function
//****************************************
Frame process_eachframe(const std::unique_ptr<tflite::Interpreter>& detection_interpreter, const std::unique_ptr<tflite::Interpreter>& movenet_interpreter,  const std::string& imgf1,  const std::string& imgf2, const std::string& output_folder)
{


    float movenet_threshold=0.2;
    float detection_threshold=0.57;
    int loop_theshold=8;
    float variance_threshold=2.5;
    int required_variance_point=9;
    double intersect_threshold=2.000001e-7;
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


    // // create model
    // std::unique_ptr<tflite::FlatBufferModel> model =
    //     tflite::FlatBufferModel::BuildFromFile("yolov8s_integer_quant.tflite");

    // //   auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
    // //   auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
    // tflite::ops::builtin::BuiltinOpResolver resolver;
    // std::unique_ptr<tflite::Interpreter> interpreter;
    // tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    // //   interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);
    // interpreter->SetAllowFp16PrecisionForFp32(false);
    // interpreter->AllocateTensors();



    // std::cout << " Tensorflow Test " << endl;

    // if (argc == 3)
    // {
    //   imgf1 = argv[1];
    //   imgf2 = argv[2];
    //   cout << imgf1 << " " << imgf2 << endl;
    // }


    //end of calling C++ tflite intrepter


    // string imgf1 ="./test/left_front5.jpg";
    // string imgf2 ="./test/right_front5.jpg";



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

    std::string stereo_file = "./stereo_cam.yml";

    // Call the function and get the camera configuration
    std::map<std::string, cv::Mat> camera_config = get_stereo_coefficients(stereo_file);

    cv::Mat imgL = cv::imread(imgf1);
    cv::Mat imgR = cv::imread(imgf2);


    // Rectify the images
    auto [rectifiedL, rectifiedR] = get_rectify_image(imgL, imgR, camera_config);


    std::vector<std::vector<float>> results1 = process_4(detection_interpreter,rectifiedL, detection_threshold);


    //*****************************************
    //start to get the 3d depth



    // // Save the rectified images
    // cv::imwrite("./rect_l.jpg", rectifiedL);
    // cv::imwrite("./rect_r.jpg", rectifiedR);
    // //********************



    // Compute the disparity map
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

    //define line segments
    std::vector<std::vector<int>> vec_inds = {
        {6, 5}, {6, 8}, {8, 10}, {5, 7}, {7, 9}, {12, 14}, {14, 16}, {11, 13}, {13, 15}, {6, 12}, {5, 11}, {12, 11}
    };


    // //prepare the intrepreter of movenet    
    // std::unique_ptr<tflite::FlatBufferModel> model_movenet=tflite::FlatBufferModel::BuildFromFile("./movenet.tflite");
    // tflite::ops::builtin::BuiltinOpResolver resolver_movenet;
    // std::unique_ptr<tflite::Interpreter> interpreter_movenet;
    // tflite::InterpreterBuilder(*model_movenet, resolver_movenet)(&interpreter_movenet);

    // interpreter_movenet->SetAllowFp16PrecisionForFp32(false);
    // interpreter_movenet->AllocateTensors();
    //end of movenet interpreter


    // std::cout << bbox_pair.size() << std::endl;

    //do movenet inference

    // printNestedVector(results1);  

    std::vector<float> box1;


    box1=findLargestBBox(results1);

    //get the biggest bounding box




    // std::cout << "print out box1" << std::endl;
    // for (float value : box1) {
    //     std::cout << value << " ";
    // }
 
    // std::cout << std::endl;


    // cv::Rect bbox1(cv::Point(box1[0], box1[1]), cv::Point(box1[2], box1[3]));

    // cv::Mat crop1=rectifiedL(bbox1);
    // Calculate crop coordinates
    int x1 = std::max(0, static_cast<int>(box1[0] - 0.05 * (box1[2] - box1[0])));
    int y1 = std::max(0, static_cast<int>(box1[1] - 0.05 * (box1[3] - box1[1])));
    int x2 = std::min(rectifiedL.cols, static_cast<int>(box1[2] + 0.05 * (box1[2] - box1[0])));
    int y2 = std::min(rectifiedL.rows, static_cast<int>(box1[3] + 0.05 * (box1[3] - box1[1])));

    // Crop and save the image
    cv::Rect cropRect(x1, y1, x2 - x1, y2 - y1);
    cv::Mat crop1 = rectifiedL(cropRect);


    std::vector<std::vector<Keypoint>> left=process_movenet_augmentation(movenet_interpreter, crop1, movenet_threshold, loop_theshold, true);



    // //draw right one on the box
    // for (const auto& point : left) {
    //     cv::Point center(static_cast<int>(point.x), static_cast<int>(point.y));
    //     cv::circle(crop1, center, 2, cv::Scalar(0, 255, 0), -1);
    // }

    // cv::imwrite("./movenet_image1.jpg", crop1);

    //from here is all correct

    // printKeypoints(left);

    //convert 2d detection points to whole image
    // x_adj, = point[0] + max(0, int(bbox[0] - 0.05 * (bbox[2] - bbox[0])))
    //y_adj =, point[1] + max(0, int(bbox[1] - 0.05 * (bbox[3] - bbox[1])))

    //do the converts to orginal image


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



        //draw the result on origin image to make sure it is correct
        

   

        

        //end of movenet

        //*****************************************
        //start to use the result of movenet to get the 3d point

        // Vector to store 3D points

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

        // printDepth3D(depth_3d);


        //get the depth of shoulder point
        cv::Vec3f sixthElement = depth_3d[5];

        // Access the third component (index 2) of the cv::Vec3f
        float thirdComponent = sixthElement[2];

        right_shoulder.push_back(thirdComponent);
        //end of testing effective range

        std::vector<float> distances = calculateDistances(depth_3d, vec_inds);

        // // Output the distances
        // for (const auto& distance : distances) {
        //     std::cout << distance << std::endl;
        // }


        // *****************************************
        // post mag processing algorithm


        // end of post processing algorithm

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
        // std::cout << "-------" << std::endl;
        // printKeypoints(left_2d);

        std::vector<Keypoint> averageKeypoints = calculateAverageKeypoints(left_2d);
        // std::cout << "-------" << std::endl;

        // for (size_t i = 0; i < averageKeypoints.size(); ++i) {
        //     std::cout << "Average Keypoint " << i << ": (" << averageKeypoints[i].x << ", " << averageKeypoints[i].y << ")" << std::endl;
        // }


        // drawLinesBetweenPoints(rectifiedL_copy, averageKeypoints, vec_inds);

        // for (const auto& point : averageKeypoints) {
        //ignore the first four points
        for (int i=5; i< averageKeypoints.size();i++){
            cv::Point center(static_cast<int>(averageKeypoints[i].x), static_cast<int>(averageKeypoints[i].y));
            cv::circle(rectifiedL_copy, center, 4, cv::Scalar(0, 255, 0), -1);

        }


        std::size_t lastSlashIndex = imgf1.find_last_of("/\\");
        std::string filename = imgf1.substr(lastSlashIndex + 1); output_folder ;

        std::string outputPath = output_folder  + filename;
        cv::imwrite(outputPath, rectifiedL_copy);






        // printNestedVector(list_of_mag);

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



        // // std::cout << "---------" << std::endl;

        // printNestedVector(variance_vector_list);


        int count=0;

        for(int i=0;i<variance_vector_list.size();i++){
            float variance = calculateVariance(variance_vector_list[i]);
            if ((variance>0)&&(variance<variance_threshold )){
                count+=1;
            }
            // else{
            //     variance_vector_list[i]={};
            // }
        }

        // std::cout << "count is: " <<count<<  std::endl;
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


namespace fs = std::filesystem;

int main(int argc, char **argv) {



    float ret_val = dist_intersect(1.0f, 1.0f, 6.0f, 1.0f);
    // std::cout << "dist_intersect: " << ret_val << std::endl;

    std::map<float, float> map_probs = gen_dict();

    double distri = 0.677f;


    //threshold for algorithm between frames
    int boundary_threshold=3;


    //call detection model and movenet model intepreter so the two interpreter won't be called twice


    //for detection model
    // create model
    std::unique_ptr<tflite::FlatBufferModel> detection_model =
        tflite::FlatBufferModel::BuildFromFile("yolov8s_integer_quant.tflite");

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
    std::unique_ptr<tflite::FlatBufferModel> movenet_model=tflite::FlatBufferModel::BuildFromFile("./movenet.tflite");
    tflite::ops::builtin::BuiltinOpResolver movenet_resolver;
    std::unique_ptr<tflite::Interpreter> movenet_interpreter;
    tflite::InterpreterBuilder(*movenet_model, movenet_resolver)(&movenet_interpreter);

    movenet_interpreter->SetAllowFp16PrecisionForFp32(false);
    movenet_interpreter->AllocateTensors();
    //end of movenet model

    std::string left_dir = "./full/left/";
    std::string right_dir = "./full/right/";
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

        // Check if the corresponding right file exists
        if (fs::exists(right_file_path)) {
            cv::Mat frameR = cv::imread(left_file_path);
            cv::Mat frameL = cv::imread(right_file_path);


            Frame frame=process_eachframe(detection_interpreter, movenet_interpreter, left_file_path, right_file_path, output_folder);
            // std::cout << "---------This frame is: " << std::endl;
            frame.printMeanAndRange();


            //check if it is empty, also add the file name
            if (!frame.isEmpty()) {
                
                valid_frames_names.push_back(file_name);

                valid_frames.push_back(frame);

            }



        }
    }

    //valid frame string
    std::cout << "Valid Frames:" << std::endl;
    for (const std::string& frame : valid_frames_names) {
        std::cout << frame << " ";
    }    

    std::cout<< "--------------" << std::endl;
    printListOfFrames(valid_frames);



    if (valid_frames.size() < 3) {
        std::cout << "no enough valid frames" << std::endl;
    } else {
        std::vector<float> adjacent_result;
        for (size_t i = 0; i < valid_frames.size() - 1; ++i) {
            adjacent_result.push_back(intersect(valid_frames[i].mean_local, valid_frames[i + 1].mean_local, valid_frames[i].range_local, valid_frames[i + 1].range_local, distri, map_probs).first);
        }

        std::cout << "Adjacent Result: ";
        for (const auto& val : adjacent_result) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::vector<size_t> indices_sorted(adjacent_result.size());
        std::iota(indices_sorted.begin(), indices_sorted.end(), 0);
        std::sort(indices_sorted.begin(), indices_sorted.end(), [&](size_t i, size_t j) { return adjacent_result[i] > adjacent_result[j]; });

        std::vector<std::pair<size_t, size_t>> pair_index;
        for (auto i : indices_sorted) {
            pair_index.push_back({i, i + 1});
        }

        std::cout << "Pair Index: ";
        for (const auto& pair : pair_index) {
            std::cout << "(" << pair.first << ", " << pair.second << ") ";
        }
        std::cout << std::endl;
    }
    


    return 0;
}




//to run this: make && ./TFLiteImageClassification


