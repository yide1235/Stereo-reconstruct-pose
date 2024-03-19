
// detection.cpp
#include "detection.hpp"


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



std::vector<std::vector<float>> detection_process(const std::unique_ptr<tflite::Interpreter>& interpreter,const cv::Mat& img, const float detection_threshold)
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

cv::Scalar getColor(int i, bool bgr ) {
    std::string hex[] = {
        "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
        "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7"
    };
    int n = sizeof(hex) / sizeof(hex[0]);
    cv::Scalar c = hex2rgb('#' + hex[i % n]);
    return bgr ? cv::Scalar(c[2], c[1], c[0]) : c;
}

cv::Mat plotOneBox(const std::vector<float>& x, cv::Mat im, cv::Scalar color, const std::string& label, int rectLineThickness, int textLineThickness) {
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



