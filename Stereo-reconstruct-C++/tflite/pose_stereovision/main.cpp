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

using namespace std;
using namespace cv;
using namespace tflite;
// ns
using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
// ns
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

typedef cv::Point3_<float> Pixel;



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

        std::cout << box[0] << " " << box[1] <<" " <<  box[2] <<" " <<  box[3] <<" " <<  std::endl;

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
  printf("Read matrix from file s: %.10f\n", elapsed_seconds.count());

  start = std::chrono::system_clock::now();

 
  cv::Mat inputImg = letterbox(img, WIDTH, HEIGHT);



  inputImg = mat_process(inputImg, WIDTH, HEIGHT);


 

  // cout << "DIM IS " << inputImg.channels() << endl;


  // cout << " Got image " << endl;

  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("Process Matrix to RGB s: %.10f\n", elapsed_seconds.count());
  interpreter->SetAllowFp16PrecisionForFp32(true);

  start = std::chrono::system_clock::now();
  // cout << " GOT INPUT IMAGE " << endl;
  
  // flatten rgb image to input layer.
  // float* input_data = interpreter->typed_input_tensor<float>(0);
  memcpy(input_tensor->data.f, inputImg.ptr<float>(0), HEIGHT*WIDTH*3* sizeof(float));


  interpreter->Invoke();
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("invoke interpreter s: %.10f\n", elapsed_seconds.count());

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
  expandBoundingBoxes(results,img.cols,img.rows, 0.03, 0.01);

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




std::vector<Keypoint> process_movenet_augmentation(const std::unique_ptr<tflite::Interpreter>& interpreter, const cv::Mat& img, float movenet_threshold, int loop_threshold) {


    
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
        std::string augmentations[3] = {"stretch","noise",  "crop"};
        std::string augmentation = augmentations[dis_int(gen)];

        cv::Mat M = cv::Mat::eye(3, 3, CV_32F);

        cv::Mat imgr_temp = img.clone();

        int imgr_height = img.rows;
        int imgr_width = img.cols;

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

        else if (augmentation == "crop") {
            int x = rand() % (imgr_width / 4);
            int y = rand() % (imgr_height / 4);
            int w = imgr_width - 2 * x;
            int h = imgr_height - 2 * y;
            cv::Rect crop_region(x, y, w, h);
            imgr_temp = imgr_temp(crop_region);
            M.at<float>(0, 2) = -x;
            M.at<float>(1, 2) = -y;
        }


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

        // Get input & output tensor
        TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);
        TfLiteTensor *output_tensor = interpreter->tensor(interpreter->outputs()[0]);

        const uint HEIGHT = input_tensor->dims->data[1];
        const uint WIDTH = input_tensor->dims->data[2];
        cv::Mat inputImg = img_input; // Assuming img is already preprocessed
        memcpy(input_tensor->data.f, inputImg.ptr<float>(0), HEIGHT * WIDTH * 3 * sizeof(float));

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

        //     // std::cout << "111111111111Keypoint (x: " << point.x << ", y: " << point.y << ")\n";
        // }
        // cv::imshow("Keypoints", imgr_temp);
        // cv::waitKey(0); // Wait for a key press to close the window


        for (int i = 0; i < points.size(); ++i) {

            auto& point = points[i];  // Get a reference to the point to modify it directly

            if(point.x != 0 && point.y != 0) {
                float x_adj = point.x;
                float y_adj = point.y;

                // Transform points back according to the inverse of M
                if (augmentation != "noise") {
                    cv::Mat inv_M = M.inv();
                    cv::Vec3f homog_point(x_adj, y_adj, 1);
                    std::vector<cv::Vec3f> transformed_points;
                    cv::transform(std::vector<cv::Vec3f>{homog_point}, transformed_points, inv_M);
                    x_adj = transformed_points[0][0];
                    y_adj = transformed_points[0][1];
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


    }//end of 50 loops


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




    //outer_list is 50 rows 17 cols
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

//end of movenet



//**********************************
//start function for triangulation
bool load_stereo_coefficients(const std::string &filename, cv::Mat &K1, cv::Mat &D1, cv::Mat &K2, cv::Mat &D2, cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F, cv::Size &imageSize) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Could not open stereo calibration file " << filename << std::endl;
        return false;
    }

    fs["K1"] >> K1;
    fs["D1"] >> D1;
    fs["K2"] >> K2;
    fs["D2"] >> D2;
    fs["R"] >> R;
    fs["T"] >> T;
    fs["E"] >> E;
    fs["F"] >> F;

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
        cv::stereoRectify(K1, D1, K2, D2, size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0.9);

        cv::initUndistortRectifyMap(K1, D1, R1, P1, size, CV_32FC1, left_map_x, left_map_y);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, size, CV_32FC1, right_map_x, right_map_y);

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
    Q.at<double>(2, 3)=922.8335512009055;
    double focal_length = Q.at<double>(2, 3);
    
    double baseline = 1.0 / Q.at<double>(3, 2);
    std::cout << "Q=\n" << Q << "\nfocal_length=" << focal_length << std::endl;
    std::cout << "T=\n" << T << "\nbaseline    =" << baseline << "mm" << std::endl;

    return config;
}

// Function to convert disparity map to 3D points
cv::Mat get_3dpoints(const cv::Mat &disparity, const cv::Mat &Q, float scale = 1.0f) {
    cv::Mat points_3d;
    cv::reprojectImageTo3D(disparity, points_3d, Q);
    points_3d = points_3d * scale;
    return points_3d;
}

// // Function to compute disparity map from left and right images
// cv::Mat get_disparity(const cv::Mat &imgL, const cv::Mat &imgR) {
//     // Placeholder for disparity calculation. Replace with your actual stereo matching algorithm.
//     // For example, you might use cv::StereoBM or cv::StereoSGBM
//     cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16, 9);
//     cv::Mat dispL;
//     stereo->compute(imgL, imgR, dispL);
//     return dispL;
// }
cv::Mat get_disparity(const cv::Mat &imgL, const cv::Mat &imgR, bool use_wls = true) {
    // StereoSGBM parameters
    int minDisparity = 0;
    int numDisparities = 5 * 16; // Must be divisible by 16
    int blockSize = 3;
    int P1 = 8 * 3 * blockSize * blockSize;
    int P2 = 32 * 3 * blockSize * blockSize;
    int disp12MaxDiff = 12;
    int uniquenessRatio = 10;
    int speckleWindowSize = 50;
    int speckleRange = 32;
    int preFilterCap = 63;
    int mode = cv::StereoSGBM::MODE_SGBM_3WAY;

    // Create StereoSGBM object
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize,
                                                            P1, P2, disp12MaxDiff, preFilterCap,
                                                            uniquenessRatio, speckleWindowSize,
                                                            speckleRange, mode);

    // Compute left disparity map
    cv::Mat dispL;
    stereo->compute(imgL, imgR, dispL);

    if (use_wls) {
        // WLS filter parameters
        double lmbda = 80000;
        double sigma = 1.3;

        // Create right matcher and compute right disparity map
        cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(stereo);
        cv::Mat dispR;
        right_matcher->compute(imgR, imgL, dispR);

        // Create and apply WLS filter
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(stereo);
        wls_filter->setLambda(lmbda);
        wls_filter->setSigmaColor(sigma);
        wls_filter->filter(dispL, imgL, dispL, dispR);
    }

    dispL.convertTo(dispL, CV_32F, 1.0 / 16);
    return dispL;
}



// Function to rectify left and right images
std::pair<cv::Mat, cv::Mat> get_rectify_image(const cv::Mat &imgL, const cv::Mat &imgR, const std::map<std::string, cv::Mat> &camera_config) {
    cv::Mat left_map_x = camera_config.at("left_map_x");
    cv::Mat left_map_y = camera_config.at("left_map_y");
    cv::Mat right_map_x = camera_config.at("right_map_x");
    cv::Mat right_map_y = camera_config.at("right_map_y");

    cv::Mat rectifiedL, rectifiedR;
    cv::remap(imgL, rectifiedL, left_map_x, left_map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(imgR, rectifiedR, right_map_x, right_map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    return {rectifiedL, rectifiedR};
}


//end of triangulation

void printKeypoints(const std::vector<Keypoint>& keypoints) {
    for (const auto& kp : keypoints) {
        std::cout << "Keypoint: (x: " << kp.x << ", y: " << kp.y << ")" << std::endl;
        // Print other properties if any
    }
}

void printDepth3Ds(const std::vector<std::vector<float>>& depth_3ds) {
    for (size_t i = 0; i < depth_3ds.size(); ++i) {
        std::cout << "Depth vector " << i << ": ";
        for (size_t j = 0; j < depth_3ds[i].size(); ++j) {
            std::cout << depth_3ds[i][j] << " ";
        }
        std::cout << std::endl;
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


int main(int argc, char **argv)
{


  float movenet_threshold=0.1;
  float detection_threshold=0.57;
  int loop_theshold=8;


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


  // create model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile("yolov8s_integer_quant.tflite");
  
  //   auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
  //   auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  //   interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);
  interpreter->SetAllowFp16PrecisionForFp32(false);
  interpreter->AllocateTensors();

  // std::cout << " Tensorflow Test " << endl;

  // if (argc == 3)
  // {
  //   imgf1 = argv[1];
  //   imgf2 = argv[2];
  //   cout << imgf1 << " " << imgf2 << endl;
  // }


  //end of calling C++ tflite intrepter


  string imgf1 ="./test/left_front5.jpg";
  string imgf2 ="./test/right_front5.jpg";



  cv::Mat img1 = cv::imread(imgf1);
  if (img1.empty()) {
      std::cerr << "Failed to load image." << std::endl;
      // You should return an empty cv::Mat or handle errors differently.
      return 0;
  }

  cv::Mat img2 = cv::imread(imgf2);
  if (img2.empty()) {
      std::cerr << "Failed to load image." << std::endl;
      // You should return an empty cv::Mat or handle errors differently.
      return 0;
  }


  std::vector<std::vector<float>> results1 = process_4(interpreter,img1, detection_threshold);
  // plotBboxes(img1, results1, coco_names, "./output1.jpg");


  std::vector<std::vector<float>> results2 = process_4(interpreter,img2, detection_threshold);
  // plotBboxes(img2, results2, coco_names, "./output2.jpg");


  // //draw the box on the image
  // plotBboxes(img1, results1, coco_names, "./output1.jpg");
  // plotBboxes(img2, results2, coco_names, "./output2.jpg");

  //end of detection model



  //*****************************************
  //start to get the 3d depth
  
  // ////start triangulation



  // // Load your left and right images (frameL and frameR) using cv::imread
  // cv::Mat frameL = cv::imread("path_to_left_image.jpg");
  // cv::Mat frameR = cv::imread("path_to_right_image.jpg");

  // // Assuming camera_config is loaded and available
  // std::map<std::string, cv::Mat> camera_config; // Load your camera_config

  std::string stereo_file = "./stereo_cam.yml";

  // Call the function and get the camera configuration
  std::map<std::string, cv::Mat> camera_config = get_stereo_coefficients(stereo_file);

  cv::Mat imgL = cv::imread(imgf1);
  cv::Mat imgR = cv::imread(imgf2);

  // Rectify the images
  auto [rectifiedL, rectifiedR] = get_rectify_image(imgL, imgR, camera_config);

  // Save the rectified images
  cv::imwrite("./rect_l.jpg", rectifiedL);
  cv::imwrite("./rect_r.jpg", rectifiedR);


  //from here proves correct
  //now get the 3d depth map



  // Compute the disparity map
  cv::Mat grayL, grayR;
  cv::cvtColor(rectifiedL, grayL, cv::COLOR_BGR2GRAY);
  cv::cvtColor(rectifiedR, grayR, cv::COLOR_BGR2GRAY);

  cv::Mat dispL = get_disparity(grayL, grayR);

  // Convert the disparity map to 3D points
  cv::Mat Q = camera_config["Q"];
  cv::Mat points_3d = get_3dpoints(dispL, Q);

  // // Split the 3D points into x, y, and depth
  // std::vector<cv::Mat> channels(3);
  // cv::split(points_3d, channels);
  // cv::Mat x = channels[0];
  // cv::Mat y = channels[1];
  // cv::Mat depth = channels[2];

  // printDepthValues(depth);



  //end of 3d depth





  //*****************************************
  //this is for movenet



  //prepare the intrepreter of movenet    
  std::unique_ptr<tflite::FlatBufferModel> model_movenet=tflite::FlatBufferModel::BuildFromFile("./movenet.tflite");
  tflite::ops::builtin::BuiltinOpResolver resolver_movenet;
  std::unique_ptr<tflite::Interpreter> interpreter_movenet;
  tflite::InterpreterBuilder(*model_movenet, resolver_movenet)(&interpreter_movenet);

  interpreter_movenet->SetAllowFp16PrecisionForFp32(false);
  interpreter_movenet->AllocateTensors();
  //end of movenet interpreter


  // std::cout << bbox_pair.size() << std::endl;
  
  //do movenet inference



  std::vector<float> box1;


  box1=results1[0];



  cv::Rect bbox1(cv::Point(box1[0], box1[1]), cv::Point(box1[2], box1[3]));

  cv::Mat crop1=img1(bbox1);


  //for right


  std::vector<Keypoint> left=process_movenet_augmentation(interpreter_movenet,crop1, movenet_threshold, loop_theshold);


  //draw right one
  for (const auto& point : left) {
      cv::Point center(static_cast<int>(point.x), static_cast<int>(point.y));
      // cv::circle(crop1, center, 2, cv::Scalar(0, 255, 0), -1);
  }

  cv::imwrite("./movenet_image1.jpg", crop1);

  // printKeypoints(left);

  //convert 2d detection points to whole image
  // x_adj, = point[0] + max(0, int(bbox[0] - 0.05 * (bbox[2] - bbox[0])))
  //y_adj =, point[1] + max(0, int(bbox[1] - 0.05 * (bbox[3] - bbox[1])))
  std::vector<Keypoint> left_converted;

  for (const auto& point : left) {
      // cv::Point center(static_cast<int>(point.x), static_cast<int>(point.y));
      // cv::circle(crop1, center, 2, cv::Scalar(0, 255, 0), -1);
      int point_x=point.x+ std::max(0, static_cast<int>(box1[0]-0.05*(box1[2]-box1[0])));
      int point_y=point.y+ std::max(0, static_cast<int>(box1[1]-0.05*(box1[3]-box1[1])));

      Keypoint temp={point_x, point_y};
      left_converted.push_back(temp);
  }

  

  //end of movenet

  //*****************************************
  //start to use the result of movenet to get the 3d point

  std::vector<std::vector<float>> depth_3ds;
  for (const auto& point : left_converted) {
    int tempx = static_cast<int>(point.x); // Set these values accordingly
    int tempy = static_cast<int>(point.y);
    tempx = std::max(0, std::min(tempx, points_3d.cols - 1));
    tempy = std::max(0, std::min(tempy, points_3d.rows - 1));

    // Access the depth value at (tempx, tempy)
    float depth_value = points_3d.at<cv::Vec3f>(tempy, tempx)[2]; // [2] accesses the z-coordinate

    // If you need to store multiple depth values
    std::vector<float> temp;
    temp.push_back(tempx);
    temp.push_back(tempy);
    temp.push_back(depth_value);

    depth_3ds.push_back(temp);
  }

  printDepth3Ds(depth_3ds);


  //*****************************************
  //post mag processing algorithm


  //end of post processing algorithm
  return 0;


}




//to run this: make && ./TFLiteImageClassification

