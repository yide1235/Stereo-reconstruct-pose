#include "opencv481.h"


#include "precomp.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "distortion_model.hpp"
#include "calib3d_c_api.h"
#include <stdio.h>
#include <iterator>



void cvStereoRectify( const CvMat* _cameraMatrix1, const CvMat* _cameraMatrix2,
                      const CvMat* _distCoeffs1, const CvMat* _distCoeffs2,
                      CvSize imageSize, const CvMat* matR, const CvMat* matT,
                      CvMat* _R1, CvMat* _R2, CvMat* _P1, CvMat* _P2,
                      CvMat* matQ, int flags, double alpha, CvSize newImgSize,
                      CvRect* roi1, CvRect* roi2 )
{
    double _om[3], _t[3] = {0}, _uu[3]={0,0,0}, _r_r[3][3], _pp[3][4];
    double _ww[3], _wr[3][3], _z[3] = {0,0,0}, _ri[3][3];
    cv::Rect_<double> inner1, inner2, outer1, outer2;

    CvMat om  = cvMat(3, 1, CV_64F, _om);
    CvMat t   = cvMat(3, 1, CV_64F, _t);
    CvMat uu  = cvMat(3, 1, CV_64F, _uu);
    CvMat r_r = cvMat(3, 3, CV_64F, _r_r);
    CvMat pp  = cvMat(3, 4, CV_64F, _pp);
    CvMat ww  = cvMat(3, 1, CV_64F, _ww); // temps
    CvMat wR  = cvMat(3, 3, CV_64F, _wr);
    CvMat Z   = cvMat(3, 1, CV_64F, _z);
    CvMat Ri  = cvMat(3, 3, CV_64F, _ri);
    double nx = imageSize.width, ny = imageSize.height;
    int i, k;

    if( matR->rows == 3 && matR->cols == 3 )
        cvRodrigues2(matR, &om);          // get vector rotation
    else
        cvConvert(matR, &om); // it's already a rotation vector
    cvConvertScale(&om, &om, -0.5); // get average rotation
    cvRodrigues2(&om, &r_r);        // rotate cameras to same orientation by averaging
    cvMatMul(&r_r, matT, &t);

    int idx = fabs(_t[0]) > fabs(_t[1]) ? 0 : 1;
    double c = _t[idx], nt = cvNorm(&t, 0, CV_L2);
    _uu[idx] = c > 0 ? 1 : -1;

    CV_Assert(nt > 0.0);

    // calculate global Z rotation
    cvCrossProduct(&t,&uu,&ww);
    double nw = cvNorm(&ww, 0, CV_L2);
    if (nw > 0.0)
        cvConvertScale(&ww, &ww, acos(fabs(c)/nt)/nw);
    cvRodrigues2(&ww, &wR);

    // apply to both views
    cvGEMM(&wR, &r_r, 1, 0, 0, &Ri, CV_GEMM_B_T);
    cvConvert( &Ri, _R1 );
    cvGEMM(&wR, &r_r, 1, 0, 0, &Ri, 0);
    cvConvert( &Ri, _R2 );
    cvMatMul(&Ri, matT, &t);

    // calculate projection/camera matrices
    // these contain the relevant rectified image internal params (fx, fy=fx, cx, cy)
    double fc_new = DBL_MAX;
    CvPoint2D64f cc_new[2] = {};

    newImgSize = newImgSize.width * newImgSize.height != 0 ? newImgSize : imageSize;
    const double ratio_x = (double)newImgSize.width / imageSize.width / 2;
    const double ratio_y = (double)newImgSize.height / imageSize.height / 2;
    const double ratio = idx == 1 ? ratio_x : ratio_y;
    fc_new = (cvmGet(_cameraMatrix1, idx ^ 1, idx ^ 1) + cvmGet(_cameraMatrix2, idx ^ 1, idx ^ 1)) * ratio;

    for( k = 0; k < 2; k++ )
    {
        const CvMat* A = k == 0 ? _cameraMatrix1 : _cameraMatrix2;
        const CvMat* Dk = k == 0 ? _distCoeffs1 : _distCoeffs2;
        CvPoint2D32f _pts[4] = {};
        CvPoint3D32f _pts_3[4] = {};
        CvMat pts = cvMat(1, 4, CV_32FC2, _pts);
        CvMat pts_3 = cvMat(1, 4, CV_32FC3, _pts_3);

        for( i = 0; i < 4; i++ )
        {
            int j = (i<2) ? 0 : 1;
            _pts[i].x = (float)((i % 2)*(nx-1));
            _pts[i].y = (float)(j*(ny-1));
        }
        cvUndistortPoints( &pts, &pts, A, Dk, 0, 0 );
        cvConvertPointsHomogeneous( &pts, &pts_3 );

        //Change camera matrix to have cc=[0,0] and fc = fc_new
        double _a_tmp[3][3];
        CvMat A_tmp  = cvMat(3, 3, CV_64F, _a_tmp);
        _a_tmp[0][0]=fc_new;
        _a_tmp[1][1]=fc_new;
        _a_tmp[0][2]=0.0;
        _a_tmp[1][2]=0.0;
        cvProjectPoints2( &pts_3, k == 0 ? _R1 : _R2, &Z, &A_tmp, 0, &pts );
        CvScalar avg = cvAvg(&pts);
        cc_new[k].x = (nx-1)/2 - avg.val[0];
        cc_new[k].y = (ny-1)/2 - avg.val[1];
    }

    // vertical focal length must be the same for both images to keep the epipolar constraint
    // (for horizontal epipolar lines -- TBD: check for vertical epipolar lines)
    // use fy for fx also, for simplicity

    // For simplicity, set the principal points for both cameras to be the average
    // of the two principal points (either one of or both x- and y- coordinates)
    if( flags & CALIB_ZERO_DISPARITY )
    {
        cc_new[0].x = cc_new[1].x = (cc_new[0].x + cc_new[1].x)*0.5;
        cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;
    }
    else if( idx == 0 ) // horizontal stereo
        cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;
    else // vertical stereo
        cc_new[0].x = cc_new[1].x = (cc_new[0].x + cc_new[1].x)*0.5;

    cvZero( &pp );
    _pp[0][0] = _pp[1][1] = fc_new;
    _pp[0][2] = cc_new[0].x;
    _pp[1][2] = cc_new[0].y;
    _pp[2][2] = 1;
    cvConvert(&pp, _P1);

    _pp[0][2] = cc_new[1].x;
    _pp[1][2] = cc_new[1].y;
    _pp[idx][3] = _t[idx]*fc_new; // baseline * focal length
    cvConvert(&pp, _P2);

    alpha = MIN(alpha, 1.);

    icvGetRectangles( _cameraMatrix1, _distCoeffs1, _R1, _P1, imageSize, inner1, outer1 );
    icvGetRectangles( _cameraMatrix2, _distCoeffs2, _R2, _P2, imageSize, inner2, outer2 );

    {
    newImgSize = newImgSize.width*newImgSize.height != 0 ? newImgSize : imageSize;
    double cx1_0 = cc_new[0].x;
    double cy1_0 = cc_new[0].y;
    double cx2_0 = cc_new[1].x;
    double cy2_0 = cc_new[1].y;
    double cx1 = newImgSize.width*cx1_0/imageSize.width;
    double cy1 = newImgSize.height*cy1_0/imageSize.height;
    double cx2 = newImgSize.width*cx2_0/imageSize.width;
    double cy2 = newImgSize.height*cy2_0/imageSize.height;
    double s = 1.;

    if( alpha >= 0 )
    {
        double s0 = std::max(std::max(std::max((double)cx1/(cx1_0 - inner1.x), (double)cy1/(cy1_0 - inner1.y)),
                            (double)(newImgSize.width - cx1)/(inner1.x + inner1.width - cx1_0)),
                        (double)(newImgSize.height - cy1)/(inner1.y + inner1.height - cy1_0));
        s0 = std::max(std::max(std::max(std::max((double)cx2/(cx2_0 - inner2.x), (double)cy2/(cy2_0 - inner2.y)),
                         (double)(newImgSize.width - cx2)/(inner2.x + inner2.width - cx2_0)),
                     (double)(newImgSize.height - cy2)/(inner2.y + inner2.height - cy2_0)),
                 s0);

        double s1 = std::min(std::min(std::min((double)cx1/(cx1_0 - outer1.x), (double)cy1/(cy1_0 - outer1.y)),
                            (double)(newImgSize.width - cx1)/(outer1.x + outer1.width - cx1_0)),
                        (double)(newImgSize.height - cy1)/(outer1.y + outer1.height - cy1_0));
        s1 = std::min(std::min(std::min(std::min((double)cx2/(cx2_0 - outer2.x), (double)cy2/(cy2_0 - outer2.y)),
                         (double)(newImgSize.width - cx2)/(outer2.x + outer2.width - cx2_0)),
                     (double)(newImgSize.height - cy2)/(outer2.y + outer2.height - cy2_0)),
                 s1);

        s = s0*(1 - alpha) + s1*alpha;
    }

    fc_new *= s;
    cc_new[0] = cvPoint2D64f(cx1, cy1);
    cc_new[1] = cvPoint2D64f(cx2, cy2);

    cvmSet(_P1, 0, 0, fc_new);
    cvmSet(_P1, 1, 1, fc_new);
    cvmSet(_P1, 0, 2, cx1);
    cvmSet(_P1, 1, 2, cy1);

    cvmSet(_P2, 0, 0, fc_new);
    cvmSet(_P2, 1, 1, fc_new);
    cvmSet(_P2, 0, 2, cx2);
    cvmSet(_P2, 1, 2, cy2);
    cvmSet(_P2, idx, 3, s*cvmGet(_P2, idx, 3));

    if(roi1)
    {
        *roi1 = cvRect(
            cv::Rect(cvCeil((inner1.x - cx1_0)*s + cx1),
                     cvCeil((inner1.y - cy1_0)*s + cy1),
                     cvFloor(inner1.width*s), cvFloor(inner1.height*s))
            & cv::Rect(0, 0, newImgSize.width, newImgSize.height)
        );
    }

    if(roi2)
    {
        *roi2 = cvRect(
            cv::Rect(cvCeil((inner2.x - cx2_0)*s + cx2),
                     cvCeil((inner2.y - cy2_0)*s + cy2),
                     cvFloor(inner2.width*s), cvFloor(inner2.height*s))
            & cv::Rect(0, 0, newImgSize.width, newImgSize.height)
        );
    }
    }

    if( matQ )
    {
        double q[] =
        {
            1, 0, 0, -cc_new[0].x,
            0, 1, 0, -cc_new[0].y,
            0, 0, 0, fc_new,
            0, 0, -1./_t[idx],
            (idx == 0 ? cc_new[0].x - cc_new[1].x : cc_new[0].y - cc_new[1].y)/_t[idx]
        };
        CvMat Q = cvMat(4, 4, CV_64F, q);
        cvConvert( &Q, matQ );
    }
}


void stereoRectify( cv::InputArray _cameraMatrix1, cv::InputArray _distCoeffs1,
                        cv::InputArray _cameraMatrix2, cv::InputArray _distCoeffs2,
                        cv::Size imageSize, cv::InputArray _Rmat, cv::InputArray _Tmat,
                        cv::OutputArray _Rmat1, cv::OutputArray _Rmat2,
                        cv::OutputArray _Pmat1, cv::OutputArray _Pmat2,
                        cv::OutputArray _Qmat, int flags,
                        double alpha, cv::Size newImageSize,
                        cv::Rect* validPixROI1, cv::Rect* validPixROI2 )
{
    cv::Mat cameraMatrix1 = _cameraMatrix1.getMat(), cameraMatrix2 = _cameraMatrix2.getMat();
    cv::Mat distCoeffs1 = _distCoeffs1.getMat(), distCoeffs2 = _distCoeffs2.getMat();
    cv::Mat Rmat = _Rmat.getMat(), Tmat = _Tmat.getMat();
    CvMat c_cameraMatrix1 = cvMat(cameraMatrix1);
    CvMat c_cameraMatrix2 = cvMat(cameraMatrix2);
    CvMat c_distCoeffs1 = cvMat(distCoeffs1);
    CvMat c_distCoeffs2 = cvMat(distCoeffs2);
    CvMat c_R = cvMat(Rmat), c_T = cvMat(Tmat);

    int rtype = CV_64F;
    _Rmat1.create(3, 3, rtype);
    _Rmat2.create(3, 3, rtype);
    _Pmat1.create(3, 4, rtype);
    _Pmat2.create(3, 4, rtype);
    cv::Mat R1 = _Rmat1.getMat(), R2 = _Rmat2.getMat(), P1 = _Pmat1.getMat(), P2 = _Pmat2.getMat(), Q;
    CvMat c_R1 = cvMat(R1), c_R2 = cvMat(R2), c_P1 = cvMat(P1), c_P2 = cvMat(P2);
    CvMat c_Q, *p_Q = 0;

    if( _Qmat.needed() )
    {
        _Qmat.create(4, 4, rtype);
        p_Q = &(c_Q = cvMat(Q = _Qmat.getMat()));
    }

    CvMat *p_distCoeffs1 = distCoeffs1.empty() ? NULL : &c_distCoeffs1;
    CvMat *p_distCoeffs2 = distCoeffs2.empty() ? NULL : &c_distCoeffs2;
    cvStereoRectify( &c_cameraMatrix1, &c_cameraMatrix2, p_distCoeffs1, p_distCoeffs2,
        cvSize(imageSize), &c_R, &c_T, &c_R1, &c_R2, &c_P1, &c_P2, p_Q, flags, alpha,
        cvSize(newImageSize), (CvRect*)validPixROI1, (CvRect*)validPixROI2);
}