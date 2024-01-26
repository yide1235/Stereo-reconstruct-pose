#ifndef OPENCV481_H
#define OPENCV481_H

#include <opencv2/opencv.hpp>

#include <opencv2/core/types_c.h>

#ifdef __cplusplus
extern "C" {
#endif

void cvStereoRectify(const CvMat* _cameraMatrix1, const CvMat* _cameraMatrix2,
                     const CvMat* _distCoeffs1, const CvMat* _distCoeffs2,
                     CvSize imageSize, const CvMat* matR, const CvMat* matT,
                     CvMat* _R1, CvMat* _R2, CvMat* _P1, CvMat* _P2,
                     CvMat* matQ, int flags, double alpha, CvSize newImgSize,
                     CvRect* roi1, CvRect* roi2);

void stereoRectify( cv::InputArray _cameraMatrix1, cv::InputArray _distCoeffs1,
                        cv::InputArray _cameraMatrix2, cv::InputArray _distCoeffs2,
                        cv::Size imageSize, cv::InputArray _Rmat, cv::InputArray _Tmat,
                        cv::OutputArray _Rmat1, cv::OutputArray _Rmat2,
                        cv::OutputArray _Pmat1, cv::OutputArray _Pmat2,
                        cv::OutputArray _Qmat, int flags,
                        double alpha, cv::Size newImageSize,
                        cv::Rect* validPixROI1, cv::Rect* validPixROI2 );

#ifdef __cplusplus
}
#endif

#endif