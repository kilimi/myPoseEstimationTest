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
	string rgbImageName = string("colorImage_A00366802050045A_0_ds.ppm");
	string depthImageName = string("depthImage_A00366802050045A_0.png");
	string xmlConfigFileName = string("defaultModule.xml");
	string calibrationFileName = string("calibration_A00366802050045A.yml");
	
	string rgbFile = inputPath + rgbImageName;
	string depthFile = inputPath + depthImageName;

	cv::Mat_<cv::Vec3b> rgb;
	cv::Mat_<int> depth;
	DescECV::Vec surf;
	
	CameraCalibrationCV cc = CameraCalibrationCV();
	cc.read(inputPath + calibrationFileName);
	//cc.printInConsole();
	
	DescriptorUtil du = DescriptorUtil();
	du.loadRGBD(rgbFile, depthFile, rgb,  depth, cc);
	
	DescriptorEstimation de  = DescriptorEstimation();
	pair<DescSeg::Vec, DescTex::Vec> temp = de.ecv(rgb, depth, surf, true, true);
	
	waitKey(0);
	return 0;
}

