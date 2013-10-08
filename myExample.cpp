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
	//string rgbImageName = string("colorImage_A00366802050045A_0_ds.ppm");
	//string depthImageName = string("depthImage_A00366802050045A_0.png");

	string rgbImageName = string("test2.png");
	string depthImageName = string("test2.exr");
	//string xmlConfigFileName = string("defaultModule.xml");
	string calibrationFileName = string("calibration_A00366802050045A.txt");
	
	string rgbFile = inputPath + rgbImageName;
	string depthFile = inputPath + depthImageName;

	cv::Mat_<cv::Vec3b> rgb;
	cv::Mat_<int> depth;
	DescECV::Vec surf;
	
	CameraCalibrationCV cc = CameraCalibrationCV::BlenderCamera(); 
	//CameraCalibrationCV cc = CameraCalibrationCV::KinectIdeal();
	//cc.addTransformationFromTxtFile(inputPath + calibrationFileName, 0);
	//cc.printInConsole();
	//cout << "-------------" << endl;
	//ck.printInConsole();
	
	DescriptorUtil().loadRGBD(rgbFile, depthFile, rgb,  depth, cc);
	
	DescRGB::Vec points;
	DescriptorUtil().reproject<DescRGB>(rgb, depth, points, CameraCalibrationCV::KinectIdeal());
	DescriptorUtil().show<DescRGB>(points);
	
	cv::Mat_<float> df = depth;
	normalize(df, df, 1, 0, CV_MINMAX);
	imshow("", df);
	waitKey();

	DescriptorEstimation de;
	pair<DescSeg::Vec, DescTex::Vec> temp = de.ecv(rgb, depth, surf, true, true);
	cout<< "me done" << endl;
	
	
	waitKey(0);
	return 0;
}

