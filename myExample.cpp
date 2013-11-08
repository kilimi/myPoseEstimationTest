#include <stdio.h>
#include "Mathematics/CameraCalibration.h"
#include <tclap/CmdLine.h>
#include <PoseEstimation/PoseEstimation.h>
#include <PoseEstimation/Descriptors.h>
#include <PoseEstimation/DescriptorEstimation.h>
#include "Features/extendedLineSegment3D.h"
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace std;
using namespace PoseEstimation;
using namespace cv;

typedef DescRGBN DescT;
void computeELSfromPointCLoud(string xmlName, string rgbFile, string depthFile, string pointCloudFile);

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
	std::vector<texlet3D> tex;
	pair<DescSeg::Vec, DescTex::Vec> temp = de.ecv(rgb, depth, surf, true, true, els, tex);
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


void justSaveElsAndTex(string xmlName, string rgbFile, string depthFile)
{
	cv::Mat_<cv::Vec3b> rgb;
	cv::Mat_<int> depth;
	DescECV::Vec surf;

	CameraCalibrationCV cc = CameraCalibrationCV::KinectIdeal();

	DescriptorUtil().loadRGBD(rgbFile, depthFile, rgb,  depth, cc);

	std::vector<extendedLineSegment3D> els;
	std::vector<texlet3D> tex;
	DescriptorEstimation de;
	pair<DescSeg::Vec, DescTex::Vec> temp = de.ecv(rgb, depth, surf, true, true, els, tex);

	string outXML;
	outXML.append("./xml/");
	outXML.append(xmlName);
	outXML.append("_ELS.xml");
	write3DExtendedLineSegmentsToXMLFile(els, outXML.c_str());

	string outXMLtex;
	outXMLtex.append("./xml/");
	outXMLtex.append(xmlName);
	outXMLtex.append("_TEX.xml");
	write3DTexletsToXMLFile(tex, outXMLtex.c_str());
	cout<< outXML << " is saved" << endl;

}



void test(string name, string inputPath, string rgbFileScene,  string depthFileScene)
{
	string _rgb;
	_rgb.append(name);
	_rgb.append(".ppm");

	string _depth;
	_depth.append(name);
	_depth.append(".png");

	string rgbFile = inputPath + _rgb;
	string depthFile = inputPath + _depth;

	cv::Mat_<cv::Vec3b> rgb;
	cv::Mat_<int> depth;
	DescECV::Vec surf;

	cv::Mat_<cv::Vec3b> rgbScene;
	cv::Mat_<int> depthScene;

	CameraCalibrationCV cc = CameraCalibrationCV::KinectIdeal();

	DescriptorUtil().loadRGBD(rgbFile, depthFile, rgb,  depth, cc);
	DescriptorUtil().loadRGBD(rgbFileScene, depthFileScene, rgbScene,  depthScene, cc);

	DescriptorEstimation de;
	std::pair<DescHist::Vec, DescHist::Vec> hist = de.histogramECV(rgb, depth, 10, 10, false, true);
	std::pair<DescHist::Vec, DescHist::Vec> histScene = de.histogramECV(rgbScene, depthScene, 50, 10, false, true);

	DescHist::Vec source, target;
	source = hist.first;
	target = histScene.first;
	//	CorrespondenceSearch<DescHist>::Ptr search = CorrespondenceSearch<DescHist>::MakeFeatureNNSearch(target, 2); //MakeXYZRadiusSearch(target, 25.0f);
	//	search->query(source);
	//	const Correspondence::Vec& corrs = search->getCorr();


	//show texlets
	//	DescriptorUtil du;
	//	DescriptorUtil::ViewT view;
	//	du.addPoints<DescHist>(view, hist.second, "tex_obj");
	//	du.addPoints<DescHist>(view, histScene.second, "tex_scn");
	//	du.show(view);

	AlignmentUtil().translate<DescHist>(source, 0, 0, -100);
	DescriptorUtil().showCorr<DescHist>(source, target, nearestFeatures<DescHist>(source, target), 50);
}


void saveLineSegments3D()
{

}



int main(int argc, char* argv[])
{
	string inputPath = string("./in/");

	string outputPath = string("./out/");
	//
	//	run("top", inputPath);
	//	run("bottom", inputPath);
	//	run("right", inputPath);
	//	run("left", inputPath);
	//	run("right2", inputPath);
	//	run("left2", inputPath);

	//	test("circle_top", inputPath, "./in/circle_rgb.tiff", "./in/circle_depth.tiff");

	//scene
	//	justSaveElsAndTex("rgb1", "./in/rgb1.tiff", "./in/depth1.tiff");

	//object
	//	justSaveElsAndTex("rc", "./in/rc_top.ppm", "./in/rc_top.png");

	//object with pointCloud

	computeELSfromPointCLoud("pointCloud", "./in/rc_top.ppm", "./in/rc_top.png", "./in/rc_pointcloud_top.png");

	waitKey(0);
	return 0;
}


void computeELSfromPointCLoud(string xmlName, string rgbFile, string depthFile, string pointCloudFile)
{
	cv::Mat_<cv::Vec3b> rgb;
	cv::Mat_<int> depth;
	DescECV::Vec surf;

	CameraCalibrationCV cc = CameraCalibrationCV::KinectIdeal();
	std::vector<extendedLineSegment3D> els;
	std::vector<texlet3D> tex;
	DescriptorEstimation de;
	DescriptorUtil().loadRGBD(rgbFile, depthFile, rgb,  depth, cc);
	pcl::PointCloud<pcl::PointXYZ> cloud; // Filled with data to copy

	if (pcl::io::loadPCDFile<pcl::PointXYZ> ("./in/top.pcd", cloud) == -1) //* load the file
	{
		PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
	}
	std::cout << "Loaded " << endl;


	const int h = cloud.height;
	const int w = cloud.width;

	cv::Mat_<cv::Vec3f> data3D(h, w, cv::Vec3f::all(0.0f));
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
			const pcl::PointXYZ& p = cloud(c,r);
			if(pcl::isFinite(p)) {
				data3D(r, c)[0] = p.x;
				data3D(r, c)[1] = p.y;
				data3D(r, c)[2] = p.z;
			}
		}
	}

//	pair<DescSeg::Vec, DescTex::Vec> temp = de.ecv(rgb, depth, data3D, surf, true, true, els, tex);

	//		string outXML;
	//		outXML.append("./xml/");
	//		outXML.append(xmlName);
	//		outXML.append("_ELS.xml");
	//		write3DExtendedLineSegmentsToXMLFile(els, outXML.c_str());
	//
	//		string outXMLtex;
	//		outXMLtex.append("./xml/");
	//		outXMLtex.append(xmlName);
	//		outXMLtex.append("_TEX.xml");
	//		write3DTexletsToXMLFile(tex, outXMLtex.c_str());
	//		cout<< outXML << " is saved" << endl;
}


