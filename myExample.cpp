#include <stdio.h>
#include "Mathematics/CameraCalibration.h"
#include <tclap/CmdLine.h>
#include <PoseEstimation/PoseEstimation.h>
#include <PoseEstimation/Descriptors.h>
#include <PoseEstimation/DescriptorEstimation.h>
#include "Features/extendedLineSegment3D.h"

using namespace std;
using namespace PoseEstimation;
using namespace cv;


void setTransformation( KVector<> &_t, KMatrix<> &_R, int camera)
{
    switch(camera)
    {
    case 0: // top
    {
        _t = (0.05, 0, 0.5);
        _R = (1, -6.12303e-17, -2.44921e-16, 6.12303e-17, 1, 6.12303e-17, 2.44921e-16, -6.12303e-17, 1);
        break;
    }
    case 1: // bottom
    {
        _t = (0.05, 0, -0.5);
        _R = (-1, 6.12303e-17, 1.22461e-16, 6.12303e-17, 1, 6.12303e-17, -1.22461e-16, 6.12303e-17, -1);

        break;
    }
    case 2: //right
    {
        _t = (0.05, 0.5, 3.06152e-18);
        _R = (1, -1.12475e-32, 0, 0, -6.12303e-17, 1, -1.2326e-32, -1, -6.12303e-17);
        break;
    }
    case 3: //left , 5th, 9th
    {
        _t[0] = 0.05;
        _t[1] = -0.5;
        _t[2] = 3.06152e-18;
        _R = (-1, 3.74915e-33, 1.22461e-16, -1.22461e-16, -6.12303e-17, -1, -1.2326e-32, -1, 6.12303e-17);
        break;
    }
    case 4: //right2
    {
        _t = (0.5, -0.05, 3.06152e-18);
        _R = (6.12303e-17, -6.12303e-17, 1, -1, 7.4983e-33, 6.12303e-17, -1.2326e-32, -1, -6.12303e-17);
        break;
    }
    case 5: //left2
    {
        _t = (-0.5, -0.05, 3.06152e-18);
        _R = (-1.83691e-16, 6.12303e-17, -1, 1, 0, -1.83691e-16, -1.2326e-32, -1, -6.12303e-17);
        break;
    }
    }
}

int main(int argc, char* argv[])
{
    string inputPath = string("./in/");
    string outputPath = string("./out/");

    string rgbImageName = string("left.ppm");
    string depthImageName = string("left.png");
    //string xmlConfigFileName = string("defaultModule.xml");
    //string calibrationFileName = string("calibration_A00366802050045A.txt");

    string rgbFile = inputPath + rgbImageName;
    string depthFile = inputPath + depthImageName;

    cv::Mat_<cv::Vec3b> rgb;
    cv::Mat_<int> depth;
    DescECV::Vec surf;

    CameraCalibrationCV cc = CameraCalibrationCV::KinectIdeal();
    //cc.addTransformationFromTxtFile(inputPath + calibrationFileName, 0);
    //cc.printInConsole();
    //cout << "-------------" << endl;
    //ck.printInConsole();

    DescriptorUtil().loadRGBD(rgbFile, depthFile, rgb,  depth, cc);

    //reproject
    //    DescRGB::Vec points;
    //    DescriptorUtil().reproject<DescRGB>(rgb, depth, points, cc);
    //    DescriptorUtil().show<DescRGB>(points);

    /*cv::Mat_<float> df = depth;
    normalize(df, df, 1, 0, CV_MINMAX);
    imshow("", df);
    waitKey();
*/
    std::vector<extendedLineSegment3D> els;
    DescriptorEstimation de;
    pair<DescSeg::Vec, DescTex::Vec> temp = de.ecv(rgb, depth, surf, true, true, els);
    KVector<> t(3);
    KMatrix<> R(3, 3);
    int camera = 3;
    setTransformation(t, R, camera);



    for (unsigned int i = 0; i < els.size(); i++)
    {
        els.at(i).rotate(R);
        els.at(i).translate(t);
    }


    write3DExtendedLineSegmentsToXMLFile(els, "leftTransformed.xml");

    cout<< "*****done*****" << endl;


    waitKey(0);
    return 0;
}

