#include <stdio.h>
#include "Mathematics/CameraCalibration.h"
#include <tclap/CmdLine.h>
#include <PoseEstimation/PoseEstimation.h>
#include <PoseEstimation/Descriptors.h>
#include <PoseEstimation/DescriptorEstimation.h>

using namespace std;
using namespace PoseEstimation;
using namespace cv;


int main(int argc, char* argv[])
{
	string inputPath = string("./in/");
	string outputPath = string("./out/");

    string rgbImageName = string("bottom.ppm");
    string depthImageName = string("bottom.png");
	//string xmlConfigFileName = string("defaultModule.xml");
	//string calibrationFileName = string("calibration_A00366802050045A.txt");
	
	string rgbFile = inputPath + rgbImageName;
	string depthFile = inputPath + depthImageName;

	cv::Mat_<cv::Vec3b> rgb;
	cv::Mat_<int> depth;
	DescECV::Vec surf;
	
	//CameraCalibrationCV cc = CameraCalibrationCV::BlenderCamera(); 
	CameraCalibrationCV cc = CameraCalibrationCV::KinectIdeal();
	//cc.addTransformationFromTxtFile(inputPath + calibrationFileName, 0);
	//cc.printInConsole();
	//cout << "-------------" << endl;
	//ck.printInConsole();
	
	DescriptorUtil().loadRGBD(rgbFile, depthFile, rgb,  depth, cc);
	
    DescRGB::Vec points;
    DescriptorUtil().reproject<DescRGB>(rgb, depth, points, cc);
    DescriptorUtil().show<DescRGB>(points);
	
	/*cv::Mat_<float> df = depth;
	normalize(df, df, 1, 0, CV_MINMAX);
	imshow("", df);
	waitKey();
*/
	DescriptorEstimation de;
	pair<DescSeg::Vec, DescTex::Vec> temp = de.ecv(rgb, depth, surf, true, true);

//    for (unsigned int i = 0; i < temp.first.size(); i++)
//    {
////      _texs3D.at(i).rotate(R);
////      _texs3D.at(i).translate(t);
//    }




	cout<< "me done" << endl;
	
	
	waitKey(0);
	return 0;
}

