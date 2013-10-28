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
	KMatrix<> yaw;
	double dataYaw[] =  {1, 0, 0, 0, -1, 0, 0, 0, -1};
	yaw.init(3,3, dataYaw);

	switch(camera)
	{
	case -1:
		cout << "No calibration data!!" << endl;
		break;
	case 0: // top
	{
		_t[0] = 0.05;
		_t[1] = 0;
		_t[2] = 0.5;
		double data[] = {1, -6.12303e-17, -2.44921e-16, 6.12303e-17, 1, 6.12303e-17, 2.44921e-16, -6.12303e-17, 1};
		_R.init(3,3,data);

		break;
	}
	case 1: // bottom
	{
		_t[0] = 0.05;
		_t[1] = 0;
		_t[2] = -0.5;
		double data[] = {-1, 6.12303e-17, 1.22461e-16, 6.12303e-17, 1, 6.12303e-17, -1.22461e-16, 6.12303e-17, -1};
		_R.init(3,3,data);
		break;
	}
	case 2: //right
	{
		_t[0] = 0.05;
		_t[1] = 0.5;
		_t[2] = 3.06152e-18;
		double data[] = {1, -1.12475e-32, 0, 0, -6.12303e-17, 1, -1.2326e-32, -1, -6.12303e-17};
		_R.init(3,3,data);
		break;
	}
	case 3: //left
	{
		_t[0] = 0.05;
		_t[1] = -0.5;
		_t[2] = 3.06152e-18;
		double data[] = {-1, 3.74915e-33, 1.22461e-16, -1.22461e-16, 6.12303e-17, -1, -1.2326e-32, -1, -6.12303e-17};
		_R.init(3,3,data);
		break;
	}
	case 4: //right2
	{
		_t[0] = 0.5;
		_t[1] = -0.05;
		_t[2] = 3.06152e-18;
		double data[] = {6.12303e-17, -6.12303e-17, 1, -1, 7.4983e-33, 6.12303e-17, -1.2326e-32, -1, -6.12303e-17};
		_R.init(3,3,data);
		break;
	}
	case 5: //left2
	{
		_t[0] = -0.5;
		_t[1] = -0.05;
		_t[2] = 3.06152e-18;
		double data[] = {-1.83691e-16, 6.12303e-17, -1, 1, 0, -1.83691e-16, -1.2326e-32, -1, -6.12303e-17};
		_R.init(3,3,data);
		break;
	}
	}

	_R = _R * yaw;
	_t= _t*1000;
}


void run(string name, string inputPath)
{
	string _rgb;
	_rgb.append(name);
	_rgb.append(".ppm");

	string _depth;
	_depth.append(name);
	_depth.append(".png");

	string rgbImageName = _rgb;
	string depthImageName = _depth;

	string rgbFile = inputPath + rgbImageName;
	string depthFile = inputPath + depthImageName;

	cv::Mat_<cv::Vec3b> rgb;
	cv::Mat_<int> depth;
	DescECV::Vec surf;

	CameraCalibrationCV cc = CameraCalibrationCV::KinectIdeal();

	DescriptorUtil().loadRGBD(rgbFile, depthFile, rgb,  depth, cc);

	//reproject
	//        DescRGB::Vec points;
	//        DescriptorUtil().reproject<DescRGB>(rgb, depth, points, cc);
	//        DescriptorUtil().show<DescRGB>(points);

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
	int camera = -1;
	if (name == "top") camera = 0;
	else if (name == "bottom") camera = 1;
	else if (name == "right") camera = 2;
	else if (name == "left") camera = 3;
	else if (name == "right2") camera = 4;
	else if (name == "left2") camera = 5;


	cout << "Camera: " << camera << endl;
	setTransformation(t, R, camera);

	for (unsigned int i = 0; i < els.size(); i++)
	{
		els.at(i).rotate(R);
		els.at(i).translate(t);
	}

	string outXML;
	outXML.append(name);
	outXML.append("Transformed.xml");
	write3DExtendedLineSegmentsToXMLFile(els, outXML.c_str());

	cout<< outXML << " is saved" << endl;
}

int main(int argc, char* argv[])
{
	string inputPath = string("./in/");

	string outputPath = string("./out/");

	run("top", inputPath);
	run("bottom", inputPath);
	run("right", inputPath);
	run("left", inputPath);
	run("right2", inputPath);
	run("left2", inputPath);

	waitKey(0);
	return 0;
}

