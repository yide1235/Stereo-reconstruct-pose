


#include "triangulation.hpp"




// //**********************************
// //start function for triangulation
// bool load_stereo_coefficients(const std::string &filename, cv::Mat &K1, cv::Mat &D1, cv::Mat &K2, cv::Mat &D2, cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F, cv::Size &imageSize) {
//     cv::FileStorage fs(filename, cv::FileStorage::READ);
//     if (!fs.isOpened()) {
//         std::cerr << "Error: Could not open stereo calibration file " << filename << std::endl;
//         return false;
//     }

//     fs["K1"] >> K1;
//     K1.convertTo(K1, CV_64F); // Ensure matrix is of type double
//     fs["D1"] >> D1;
//     D1.convertTo(D1, CV_64F); // Ensure matrix is of type double
//     fs["K2"] >> K2;
//     K2.convertTo(K2, CV_64F); // Ensure matrix is of type double
//     fs["D2"] >> D2;
//     D2.convertTo(D2, CV_64F); // Ensure matrix is of type double
//     fs["R"] >> R;
//     R.convertTo(R, CV_64F); // Ensure matrix is of type double
//     fs["T"] >> T;
//     T.convertTo(T, CV_64F); // Ensure matrix is of type double
//     fs["E"] >> E;
//     E.convertTo(E, CV_64F); // Ensure matrix is of type double
//     fs["F"] >> F;
//     F.convertTo(F, CV_64F); // Ensure matrix is of type double

//     cv::Mat sizeMat;
//     fs["size"] >> sizeMat;
//     imageSize = cv::Size(static_cast<int>(sizeMat.at<double>(0)), static_cast<int>(sizeMat.at<double>(1)));

//     fs.release();
//     return true;
// }




bool load_stereo_coefficients(cv::Mat &K1, cv::Mat &D1, cv::Mat &K2, cv::Mat &D2, cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F, cv::Size &imageSize, cv::Mat &R1, cv::Mat &R2, cv::Mat &P1, cv::Mat &P2) {
    // Directly setting the imageSize
    imageSize = cv::Size(1920, 1080);

    // Directly setting the matrices
    K1 = (cv::Mat_<double>(3, 3) << 1463.1431221075463, 0.0, 946.29655753964482,
                                   0.0, 1463.0557791080048, 525.3012918412893,
                                   0.0, 0.0, 1.0);
    D1 = (cv::Mat_<double>(1, 5) << 0.033055274896744626, -0.14057169795664667,
                                    -0.0042079102381219107, -0.0068009041018483466,
                                     0.50099505005713119);
    K2 = (cv::Mat_<double>(3, 3) << 1447.0281956468018, 0.0, 1006.6575774110983,
                                   0.0, 1443.2309948850107, 527.52870356505537,
                                   0.0, 0.0, 1.0);
    D2 = (cv::Mat_<double>(1, 5) << -0.0068048386700675090, 0.69346706645455880,
                                    -0.0040805311790399210, 0.021458088840686260,
                                    -3.4206917189074955);
    R = (cv::Mat_<double>(3, 3) << 0.99999631407705014, 0.00089218672794160890,
                                   -0.0025643391265946841, -0.00092520499540045922,
                                    0.99991631547435211, -0.012903722095291842,
                                    0.0025526120014968149, 0.012906047072536257,
                                    0.99991345531547449);
    T = (cv::Mat_<double>(3, 1) << -66.341287748459379, 0.89210281640217348,
                                    -10.062773201457061);
    E = (cv::Mat_<double>(3, 3) << -0.0070329356778526470, 10.073444623997023,
                                    0.76217838074581146, -9.8933925435489076,
                                    0.84722590983766310, 66.361350625682718,
                                    -0.83072023735375500, -66.336531931556038,
                                    0.85833719470695469);
    F = (cv::Mat_<double>(3, 3) << -4.6648889841947499e-09, 6.6820327042127304e-06,
                                    -0.0027659788435589125, -6.5794721767472492e-06,
                                     5.6347022084445707e-07, 0.070502675815116850,
                                     0.0026782294092082635, -0.070697504134509911, 1.0);

    R1 = (cv::Mat_<double>(3, 3) << 9.8899513056583044e-01, -1.0475489742660323e-02,
                                    1.4757674556557776e-01, 1.1423590479496794e-02,
                                     9.9991918862838236e-01, -5.5783324851059182e-03,
                                    -1.4750638392162088e-01, 7.2027999700849986e-03,
                                    9.8903487621769848e-01 );

    R2 = (cv::Mat_<double>(3, 3) << 9.8860370229023509e-01, -1.3293925654001493e-02,
                                    1.4995329725866638e-01, 1.2329967738383298e-02,
                                    9.9989692295456167e-01, 7.3563144012436668e-03,
                                    -1.5003563481256882e-01, -5.4235603348169025e-03,
                                    9.8866571361592381e-01);

    P1 = (cv::Mat_<double>(3, 3) << 9.2283355120090550e+02, 0., 6.6143807220458984e+02, 0., 0.,
       9.2283355120090550e+02, 5.2272535324096680e+02, 0., 0., 0., 1.,
       0. );

    P2 = (cv::Mat_<double>(3, 3) << 9.2283355120090550e+02, 0., 6.6143807220458984e+02,
       -6.1927712815886553e+04, 0., 9.2283355120090550e+02,
       5.2272535324096680e+02, 0., 0., 0., 1., 0.);

    return true;
}








std::map<std::string, cv::Mat> get_stereo_coefficients(bool rectify) {
    cv::Mat K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q;
    cv::Size size;

    if (!load_stereo_coefficients( K1, D1, K2, D2, R, T, E, F, size, R1, R2, P1, P2)) {
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
    config["R1"] =R1;
    config["R2"] = R2;
    config["P1"] = P1;
    config["P2"] = P2;






    if (rectify) {
        cv::Mat left_map_x, left_map_y, right_map_x, right_map_y;


        //***********have to use opencv4.8.1 for now, opencv4.5 gives different results!!!!!!!!!!
        stereoRectify(K1, D1, K2, D2, size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0.9);


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





cv::Mat get_3dpoints(const cv::Mat& disparity, const cv::Mat& Q, float scale) {
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
cv::Mat get_filter_disparity( cv::Mat& imgL,  cv::Mat& imgR, bool use_wls ) {
    

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



void printDepthValues(const cv::Mat& depth, int numRowsToPrint, int numColsToPrint) {
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

