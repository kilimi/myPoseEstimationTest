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
std::pair<DescHist::Vec, DescHist::Vec> computeELSfromPointCLoud(string xmlName, string rgbFile, string depthFile, string pointCloudFile, bool usePCD);
std::pair<DescSeg::Vec, DescTex::Vec>  computeELSfromPointCLoud2(string xmlName, string rgbFile, string depthFile, string pointCloudFile, bool usePCD);
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

//get depth and rgb image from point cloud
Mat_<ushort> getDepthMapFromPointCloud(pcl::PointCloud<pcl::PointXYZRGBA> cloud){
	Mat_<ushort> depthm(cloud.height, cloud.width);
	for(size_t r = 0; r < depthm.rows; ++r)
	{
		for(size_t c = 0; c < depthm.cols; ++c)
		{
			depthm[r][c] = (ushort)(cloud(c,r).z * (-1000));
		}
	}

	return depthm;
}


cv::Mat getRgbImageFromPointCloud(pcl::PointCloud<pcl::PointXYZRGBA> cloud){
	cv::Mat result;
	if (cloud.isOrganized()) {
		result = cv::Mat(cloud.height, cloud.width, CV_8UC3);

		if (!cloud.empty()) {

			for (int h=0; h<result.rows; h++) {
				for (int w=0; w<result.cols; w++) {
					pcl::PointXYZRGBA point = cloud.at(w, h);

					Eigen::Vector3i rgb = point.getRGBVector3i();

					result.at<cv::Vec3b>(h,w)[0] = rgb[2];
					result.at<cv::Vec3b>(h,w)[1] = rgb[1];
					result.at<cv::Vec3b>(h,w)[2] = rgb[0];
				}
			}
		}
	}
	return result;
}

//*************************************************************************************************************

std::pair<DescSeg::Vec, DescTex::Vec> getSegAndTex(string xmlName, cv::Mat_<cv::Vec3b> rgb,
		cv::Mat_<int> depth, string pointCloudFile, bool usePCD){
	CameraCalibrationCV cc = CameraCalibrationCV::KinectIdeal();

	DescECV::Vec surf;

	const int w = rgb.cols;
	const int h = rgb.rows;

	pcl::PointCloud<pcl::PointXYZ> cloud;

	if (pcl::io::loadPCDFile<pcl::PointXYZ> (pointCloudFile, cloud) == -1)
	{
		std::cout << "Couldn't read file pcd" << endl;
	}
	else std::cout << "Loaded " << endl;

	cout << "is cloud organised? " << cloud.isOrganized() << endl;
	cout << "cloud height: " << cloud.height << " , width: " << cloud.width << endl;
	const int hc = cloud.height;
	const int wc = cloud.width;


	//check if it is artificial data or real
	int artificial = 1;
	for (int r = 0; r < hc; ++r) {
		for (int c = 0; c < wc; ++c) {
			const pcl::PointXYZ& p = cloud(c,r);
			if(pcl::isFinite(p)) {
				if (p.z < 0)
				{
					artificial = -1;
					break;
				}
			}
		}
	}


	//converts from pcd to Mat
	cv::Mat_<cv::Vec3f> data3D(hc, wc, cv::Vec3f::all(0.0f));
	for (int r = 0; r < hc; ++r) {
		for (int c = 0; c < wc; ++c) {
			const pcl::PointXYZ& p = cloud(c,r);
			if(pcl::isFinite(p)) {
				data3D(r, c)[0] = p.x * 1000;
				data3D(r, c)[1] = p.y * 1000;
				data3D(r, c)[2] = p.z * 1000 * artificial;
			}
		}
	}


	DescXYZ::Vec points;
	cv::Mat_<int> map;
	DescriptorUtil().reproject(depth, points, map, cc);

	const int size = points.size();

	// Check the mapping
	cv::Mat_<int> mapp;
	if (map.empty()) { // If empty, default to dense map, row-major
		COVIS_ASSERT_MSG(size == w*h, "Too few 3D points!");;

		mapp = cv::Mat_<int>(h, w);
		int index = 0;
		for (int r = 0; r < h; ++r)
			for (int c = 0;  c < w; ++c)
				mapp(r, c) = index++;
	} else {
		mapp = map;
	}



	DescriptorEstimation de;

	//pair<DescSeg::Vec, DescTex::Vec> temp =de.ecv(rgb, depth, surf, true, true, els, tex);
	//de.getElsAndTex(rgb, depth, data3D, surf, els, tex);


	std::tr1::shared_ptr<std::vector<extendedLineSegment3D> > _els(new std::vector<extendedLineSegment3D>);
	std::tr1::shared_ptr<std::vector<lineSegment2D> > _ls(new std::vector<lineSegment2D>);

	// Kinect feature module
	Modules::KinectFeatureModule kfm;

	// Module parameters
	XMLWrapper::XMLNode rootnode;
	if (XMLWrapper::getXMLRootNode("module.xml", rootnode)) {
		kfm.setParametersFromXML(rootnode, "kinectFeatures");
	} else {
		std::cerr << "Failed to load module configuration file \"module.xml\"!" << std::endl;
		std::cerr << "\tSetting default extraction parameters..." << std::endl;
		// Set some default parameters
		kfm.useRansac(50, 1.8f);
	}

	// Calibration
	kfm.setCalibration(cc, w, h); // TODO

	// Compute image of XYZ values
	cv::Mat_<cv::Vec3f> data3D2(h, w, cv::Vec3f::all(0.0f));
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
			const int idx = map(r, c);
			if (idx != -1) {
				const DescXYZ& d = points[idx];
				data3D2(r, c)[0] = d.x;
				data3D2(r, c)[1] = d.y;
				data3D2(r, c)[2] = d.z;
			}
		}
	}

	// Initialize
	COVIS_ASSERT(kfm.Init());

	// Set input data
	cv::Mat imgBGR; // NOTE: This assumes BGR alignment!
	cv::cvtColor(rgb, imgBGR, CV_RGB2BGR);
	kfm.setKinectData(depth, imgBGR);

	// Results
	std::pair<DescSeg::Vec, DescTex::Vec> result;
	std::vector<int> keypointsSeg;
	std::vector<int> keypointsTex;
	keypointsSeg.clear();
	keypointsTex.clear();

	DescXYZ::Vec _temp = DescriptorUtil().fromPCL<pcl::PointXYZ, DescXYZ>(cloud, false);
	// Search structure in input
	typename CorrespondenceSearch<DescXYZ>::Ptr searchInput = CorrespondenceSearch<DescXYZ>::MakeXYZNNSearch(_temp, 1);


	if (!usePCD) kfm.computeExtendedLineSegments(_els, _ls, data3D2);
	else kfm.computeExtendedLineSegments(_els, _ls, data3D);


	string outXML;
	outXML.append("./xml/");
	outXML.append(xmlName);
	outXML.append("_ELS.xml");

	write3DExtendedLineSegmentsToXMLFile(*_els, outXML.c_str());


	//save temp images
	const_cast<SingleView<double>&>(kfm.getSingleView()).saveImages();
	// Store segments
	result.first = DescriptorEstimation().eseg<DescSeg>(*_els);
	keypointsSeg.reserve(result.first.size());
	for (DescSeg::Vec::const_iterator it = result.first.begin(); it != result.first.end(); ++it) {
		searchInput->query(*it);
		const int idx3D = searchInput->getNearestIdx().front();
		keypointsSeg.push_back(idx3D);
	}

	COVIS_ASSERT_MSG(!_ls->empty(), "No 2D line segments found!");
	COVIS_ASSERT_MSG(!_els->empty(), "No extended 3D line segments found!");


	//********************TEXLETS*******************************************************

	//texlets
	std::tr1::shared_ptr<std::vector<texlet2D> > texlet2Ds(new std::vector<texlet2D>);
	std::tr1::shared_ptr<std::vector<texlet3D> > texlet3Ds(new std::vector<texlet3D>);
	if (!usePCD) kfm.computeTexlets(texlet3Ds, texlet2Ds, data3D2, false);
	else kfm.computeTexlets(texlet3Ds, texlet2Ds, data3D, false);

	// Store texlets
	result.second = DescriptorEstimation().tex<DescTex>(*texlet3Ds);
	std::vector<texlet3D> texs = *texlet3Ds;
	// TODO: Keypoint: nearest input point
	keypointsTex.reserve(result.second.size());
	for (DescTex::Vec::const_iterator it = result.second.begin(); it != result.second.end(); ++it) {
		searchInput->query(*it);
		const int idx3D = searchInput->getNearestIdx().front();
		keypointsTex.push_back(idx3D);
	}

	// Check
	COVIS_ASSERT_MSG(!texlet2Ds->empty(), "No 2D texlets found!");
	COVIS_ASSERT_MSG(!texlet3Ds->empty(), "No 3D texlets found!");
	string outXMLtex;
	outXMLtex.append("./xml/");
	outXMLtex.append(xmlName);
	outXMLtex.append("_TEX.xml");
	write3DTexletsToXMLFile(*texlet3Ds, outXMLtex.c_str());

	//*************SURFACE*******************************************
	// TODO: Surface: uniform texlets with highest possible density
	Modules::KinectFeatureModule kfmSurf;
	kfmSurf.setCalibration(cc, w, h); // TODO
	kfmSurf.setFilterType(MONOGENIC_FREQ0110);
	kfmSurf.setTexletiDThresholds(0.01, 0.01);
	kfmSurf.setKinectData(depth, imgBGR);

	// Initialize
	COVIS_ASSERT(kfmSurf.Init());

	// Run
	std::tr1::shared_ptr<std::vector<texlet2D> > texlet2DSurf(new std::vector<texlet2D>);
	std::tr1::shared_ptr<std::vector<texlet3D> > texlet3DSurf(new std::vector<texlet3D>);

	if (!usePCD) kfmSurf.computeTexlets(texlet3DSurf, texlet2DSurf, data3D2, true);
	else kfmSurf.computeTexlets(texlet3DSurf, texlet2DSurf, data3D, true);

	DescECV::Vec surface = DescriptorEstimation().tex<DescECV>(*texlet3DSurf);

	bool _normalize = true;
	// Normalization
	if (_normalize) {
		normalize<DescECV>(surface);
		normalize<DescSeg>(result.first);
		normalize<DescTex>(result.second);
	} else {
		std::cerr << "Warning: normalization disabled during descriptor estimation!" << std::endl;
		std::cerr << "\tTo use the context descriptors for further processing such as correspondence search, remember to normalize first!" << std::endl;
	}

	return result;
}




int main(int argc, char* argv[])
{
	string inputPath = string("./in/");
	string outputPath = string("./out/");

	DescTex::Vec texobj, texscn;

	bool flag = true;

			std::pair<DescSeg::Vec, DescTex::Vec> obj = computeELSfromPointCLoud2("object", "./in/top.ppm", "./in/top.png", "./in/top.pcd", flag);
	//		std::pair<DescSeg::Vec, DescTex::Vec> scene = computeELSfromPointCLoud2("s2", "./in/s2.tiff", "./in/s2_depth.tiff", "./in/scene2.pcd", flag);

	string pointcl = "./in/top.pcd";
	pcl::PointCloud<pcl::PointXYZRGBA> cloud;

	if (pcl::io::loadPCDFile<pcl::PointXYZRGBA> ("./in/scene2.pcd", cloud) == -1)
	{
		std::cout << "Couldn't read file pcd" << endl;
	}
	else std::cout << "Loaded " << endl;

	//	cv::Mat_<cv::Vec3b> rgb;
	//	cv::Mat_<int> depth;
	//	CameraCalibrationCV cc = CameraCalibrationCV::KinectIdeal();
	//	DescriptorUtil().loadRGBD("./in/top.ppm", "./in/s2_depth.tiff", rgb,  depth, cc);

	Mat_<ushort> depthImg = getDepthMapFromPointCloud(cloud);
	//	cv::Mat_<float> df = depthImg;
	//	normalize(df, df, 1, 0, CV_MINMAX);
	//	imshow("", df);
	//
	//	cv::Mat_<float> df2 = depth;
	//	normalize(df2, df2, 1, 0, CV_MINMAX);
	//	imshow("d2", df2);

	Mat_<Vec3b> rgb2 = getRgbImageFromPointCloud(cloud);


	std::pair<DescSeg::Vec, DescTex::Vec> scene = getSegAndTex("scene2Res", rgb2, depthImg, pointcl, flag);


//
//		texobj = obj.second;
//		texscn = scene.second;
//
//		//	DescriptorUtil().show<DescTex>(texobj, texscn, "crap");
//
//		Recognition<DescTex,DescHist>::Ptr rec(new RecognitionVoting<DescTex,DescHist>(1, 25, 5, 0.05, false, false, false, INF_FLOAT));
//		//	rec->setVerbose(true);
//		rec->setCoplanarityFraction(1);
//
//		rec->loadObjectsL(std::vector<DescTex::Vec>(1,texobj));
//
//
//		DescriptorUtil().showDetections<DescTex>(std::vector<DescTex::Vec>(1,texobj), texscn, rec->recognizeL(texscn));
//

	waitKey(0);
	return 0;
}

std::pair<DescHist::Vec, DescHist::Vec> computeELSfromPointCLoud(string xmlName, string rgbFile, string depthFile, string pointCloudFile, bool usePCD){
	CameraCalibrationCV cc = CameraCalibrationCV::KinectIdeal();

	cv::Mat_<cv::Vec3b> rgb;
	cv::Mat_<int> depth;
	DescECV::Vec surf;

	DescriptorUtil().loadRGBD(rgbFile, depthFile, rgb,  depth, cc);
	const int w = rgb.cols;
	const int h = rgb.rows;


	pcl::PointCloud<pcl::PointXYZ> cloud;

	if (pcl::io::loadPCDFile<pcl::PointXYZ> (pointCloudFile, cloud) == -1)
	{
		std::cout << "Couldn't read file pcd" << endl;
	}
	else std::cout << "Loaded " << endl;

	cout << "is cloud organised? " << cloud.isOrganized() << endl;
	cout << "cloud height: " << cloud.height << " , width: " << cloud.width << endl;
	const int hc = cloud.height;
	const int wc = cloud.width;


	//check if it is artificial data or real
	int artificial = 1;
	for (int r = 0; r < hc; ++r) {
		for (int c = 0; c < wc; ++c) {
			const pcl::PointXYZ& p = cloud(c,r);
			if(pcl::isFinite(p)) {
				if (p.z < 0)
				{
					artificial = -1;
					break;
				}
			}
		}
	}


	//converts from pcd to Mat
	cv::Mat_<cv::Vec3f> data3D(hc, wc, cv::Vec3f::all(0.0f));
	for (int r = 0; r < hc; ++r) {
		for (int c = 0; c < wc; ++c) {
			const pcl::PointXYZ& p = cloud(c,r);
			if(pcl::isFinite(p)) {
				data3D(r, c)[0] = p.x * 1000;
				data3D(r, c)[1] = p.y * 1000;
				data3D(r, c)[2] = p.z * 1000 * artificial;
			}
		}
	}


	DescXYZ::Vec points;
	cv::Mat_<int> map;
	DescriptorUtil().reproject(depth, points, map, cc);

	const int size = points.size();

	// Check the mapping
	cv::Mat_<int> mapp;
	if (map.empty()) { // If empty, default to dense map, row-major
		COVIS_ASSERT_MSG(size == w*h, "Too few 3D points!");;

		mapp = cv::Mat_<int>(h, w);
		int index = 0;
		for (int r = 0; r < h; ++r)
			for (int c = 0;  c < w; ++c)
				mapp(r, c) = index++;
	} else {
		mapp = map;
	}



	DescriptorEstimation de;

	//pair<DescSeg::Vec, DescTex::Vec> temp =de.ecv(rgb, depth, surf, true, true, els, tex);
	//de.getElsAndTex(rgb, depth, data3D, surf, els, tex);


	std::tr1::shared_ptr<std::vector<extendedLineSegment3D> > _els(new std::vector<extendedLineSegment3D>);
	std::tr1::shared_ptr<std::vector<lineSegment2D> > _ls(new std::vector<lineSegment2D>);

	// Kinect feature module
	Modules::KinectFeatureModule kfm;

	// Module parameters
	XMLWrapper::XMLNode rootnode;
	if (XMLWrapper::getXMLRootNode("module.xml", rootnode)) {
		kfm.setParametersFromXML(rootnode, "kinectFeatures");
	} else {
		std::cerr << "Failed to load module configuration file \"module.xml\"!" << std::endl;
		std::cerr << "\tSetting default extraction parameters..." << std::endl;
		// Set some default parameters
		kfm.useRansac(50, 1.8f);
	}

	// Calibration
	kfm.setCalibration(cc, w, h); // TODO

	// Compute image of XYZ values
	cv::Mat_<cv::Vec3f> data3D2(h, w, cv::Vec3f::all(0.0f));
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
			const int idx = map(r, c);
			if (idx != -1) {
				const DescXYZ& d = points[idx];
				data3D2(r, c)[0] = d.x;
				data3D2(r, c)[1] = d.y;
				data3D2(r, c)[2] = d.z;
			}
		}
	}

	// Initialize
	COVIS_ASSERT(kfm.Init());

	// Set input data
	cv::Mat imgBGR; // NOTE: This assumes BGR alignment!
	cv::cvtColor(rgb, imgBGR, CV_RGB2BGR);
	kfm.setKinectData(depth, imgBGR);

	// Results
	std::pair<DescSeg::Vec, DescTex::Vec> result;
	std::vector<int> keypointsSeg;
	std::vector<int> keypointsTex;
	keypointsSeg.clear();
	keypointsTex.clear();

	DescXYZ::Vec _temp = DescriptorUtil().fromPCL<pcl::PointXYZ, DescXYZ>(cloud, false);
	// Search structure in input
	typename CorrespondenceSearch<DescXYZ>::Ptr searchInput = CorrespondenceSearch<DescXYZ>::MakeXYZNNSearch(_temp, 1);


	if (!usePCD) kfm.computeExtendedLineSegments(_els, _ls, data3D2);
	else kfm.computeExtendedLineSegments(_els, _ls, data3D);

	//save temp images
	const_cast<SingleView<double>&>(kfm.getSingleView()).saveImages();
	// Store segments
	result.first = DescriptorEstimation().eseg<DescSeg>(*_els);
	keypointsSeg.reserve(result.first.size());
	for (DescSeg::Vec::const_iterator it = result.first.begin(); it != result.first.end(); ++it) {
		searchInput->query(*it);
		const int idx3D = searchInput->getNearestIdx().front();
		keypointsSeg.push_back(idx3D);
	}

	COVIS_ASSERT_MSG(!_ls->empty(), "No 2D line segments found!");
	COVIS_ASSERT_MSG(!_els->empty(), "No extended 3D line segments found!");

	string outXML;
	outXML.append("./xml/");
	outXML.append(xmlName);
	outXML.append("_ELS.xml");

	KVector<> _t(3);
	_t[0] = 0;
	_t[1] = 0;
	_t[2] = 0;
	//	_t[1] = 3;
	//	_t[2] = 15;

	std::vector<extendedLineSegment3D> els1;
	els1 = *_els;
	//	for (unsigned int i = 0; i < els1.size(); i++)
	//	{
	//		els1.at(i).translate(_t);
	//	}

	write3DExtendedLineSegmentsToXMLFile(els1, outXML.c_str());

	//********************TEXLETS*******************************************************

	//texlets
	std::tr1::shared_ptr<std::vector<texlet2D> > texlet2Ds(new std::vector<texlet2D>);
	std::tr1::shared_ptr<std::vector<texlet3D> > texlet3Ds(new std::vector<texlet3D>);
	if (!usePCD) kfm.computeTexlets(texlet3Ds, texlet2Ds, data3D2, false);
	else kfm.computeTexlets(texlet3Ds, texlet2Ds, data3D, false);

	// Store texlets
	result.second = DescriptorEstimation().tex<DescTex>(*texlet3Ds);
	std::vector<texlet3D> texs = *texlet3Ds;
	// TODO: Keypoint: nearest input point
	keypointsTex.reserve(result.second.size());
	for (DescTex::Vec::const_iterator it = result.second.begin(); it != result.second.end(); ++it) {
		searchInput->query(*it);
		const int idx3D = searchInput->getNearestIdx().front();
		keypointsTex.push_back(idx3D);
	}

	// Check
	COVIS_ASSERT_MSG(!texlet2Ds->empty(), "No 2D texlets found!");
	COVIS_ASSERT_MSG(!texlet3Ds->empty(), "No 3D texlets found!");

	string outXMLtex;
	outXMLtex.append("./xml/");
	outXMLtex.append(xmlName);
	outXMLtex.append("_TEX.xml");
	write3DTexletsToXMLFile(*texlet3Ds, outXMLtex.c_str());

	//*************SURFACE*******************************************
	// TODO: Surface: uniform texlets with highest possible density
	Modules::KinectFeatureModule kfmSurf;
	kfmSurf.setCalibration(cc, w, h); // TODO
	kfmSurf.setFilterType(MONOGENIC_FREQ0110);
	kfmSurf.setTexletiDThresholds(0.01, 0.01);
	kfmSurf.setKinectData(depth, imgBGR);

	// Initialize
	COVIS_ASSERT(kfmSurf.Init());

	// Run
	std::tr1::shared_ptr<std::vector<texlet2D> > texlet2DSurf(new std::vector<texlet2D>);
	std::tr1::shared_ptr<std::vector<texlet3D> > texlet3DSurf(new std::vector<texlet3D>);

	if (!usePCD) kfmSurf.computeTexlets(texlet3DSurf, texlet2DSurf, data3D2, true);
	else kfmSurf.computeTexlets(texlet3DSurf, texlet2DSurf, data3D, true);

	DescECV::Vec surface = DescriptorEstimation().tex<DescECV>(*texlet3DSurf);

	bool _normalize = true;
	// Normalization
	if (_normalize) {
		normalize<DescECV>(surface);
		normalize<DescSeg>(result.first);
		normalize<DescTex>(result.second);
	} else {
		std::cerr << "Warning: normalization disabled during descriptor estimation!" << std::endl;
		std::cerr << "\tTo use the context descriptors for further processing such as correspondence search, remember to normalize first!" << std::endl;
	}

	//*********************************histogram

	std::pair<DescHist::Vec, DescHist::Vec> hist = DescriptorEstimation().histogramECV(result, surface, 10, 10, false, true);

	return hist;
}

std::pair<DescSeg::Vec, DescTex::Vec>  computeELSfromPointCLoud2(string xmlName, string rgbFile, string depthFile, string pointCloudFile, bool usePCD)
																								{
	CameraCalibrationCV cc = CameraCalibrationCV::KinectIdeal();

	cv::Mat_<cv::Vec3b> rgb;
	cv::Mat_<int> depth;
	DescECV::Vec surf;

	DescriptorUtil().loadRGBD(rgbFile, depthFile, rgb,  depth, cc);
	const int w = rgb.cols;
	const int h = rgb.rows;

	pcl::PointCloud<pcl::PointXYZ> cloud;

	if (pcl::io::loadPCDFile<pcl::PointXYZ> (pointCloudFile, cloud) == -1)
	{
		std::cout << "Couldn't read file pcd" << endl;
	}
	else std::cout << "Loaded " << endl;

	cout << "is cloud organised? " << cloud.isOrganized() << endl;
	cout << "cloud height: " << cloud.height << " , width: " << cloud.width << endl;
	const int hc = cloud.height;
	const int wc = cloud.width;


	//check if it is artificial data or real
	int artificial = 1;
	for (int r = 0; r < hc; ++r) {
		for (int c = 0; c < wc; ++c) {
			const pcl::PointXYZ& p = cloud(c,r);
			if(pcl::isFinite(p)) {
				if (p.z < 0)
				{
					artificial = -1;
					break;
				}
			}
		}
	}


	//converts from pcd to Mat
	cv::Mat_<cv::Vec3f> data3D(hc, wc, cv::Vec3f::all(0.0f));
	for (int r = 0; r < hc; ++r) {
		for (int c = 0; c < wc; ++c) {
			const pcl::PointXYZ& p = cloud(c,r);
			if(pcl::isFinite(p)) {
				data3D(r, c)[0] = p.x * 1000;
				data3D(r, c)[1] = p.y * 1000;
				data3D(r, c)[2] = p.z * 1000 * artificial;
			}
		}
	}


	DescXYZ::Vec points;
	cv::Mat_<int> map;
	DescriptorUtil().reproject(depth, points, map, cc);

	const int size = points.size();

	// Check the mapping
	cv::Mat_<int> mapp;
	if (map.empty()) { // If empty, default to dense map, row-major
		COVIS_ASSERT_MSG(size == w*h, "Too few 3D points!");;

		mapp = cv::Mat_<int>(h, w);
		int index = 0;
		for (int r = 0; r < h; ++r)
			for (int c = 0;  c < w; ++c)
				mapp(r, c) = index++;
	} else {
		mapp = map;
	}



	DescriptorEstimation de;

	//pair<DescSeg::Vec, DescTex::Vec> temp =de.ecv(rgb, depth, surf, true, true, els, tex);
	//de.getElsAndTex(rgb, depth, data3D, surf, els, tex);


	std::tr1::shared_ptr<std::vector<extendedLineSegment3D> > _els(new std::vector<extendedLineSegment3D>);
	std::tr1::shared_ptr<std::vector<lineSegment2D> > _ls(new std::vector<lineSegment2D>);

	// Kinect feature module
	Modules::KinectFeatureModule kfm;

	// Module parameters
	XMLWrapper::XMLNode rootnode;
	if (XMLWrapper::getXMLRootNode("module.xml", rootnode)) {
		kfm.setParametersFromXML(rootnode, "kinectFeatures");
	} else {
		std::cerr << "Failed to load module configuration file \"module.xml\"!" << std::endl;
		std::cerr << "\tSetting default extraction parameters..." << std::endl;
		// Set some default parameters
		kfm.useRansac(50, 1.8f);
	}

	// Calibration
	kfm.setCalibration(cc, w, h); // TODO

	// Compute image of XYZ values
	cv::Mat_<cv::Vec3f> data3D2(h, w, cv::Vec3f::all(0.0f));
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
			const int idx = map(r, c);
			if (idx != -1) {
				const DescXYZ& d = points[idx];
				data3D2(r, c)[0] = d.x;
				data3D2(r, c)[1] = d.y;
				data3D2(r, c)[2] = d.z;
			}
		}
	}

	// Initialize
	COVIS_ASSERT(kfm.Init());

	// Set input data
	cv::Mat imgBGR; // NOTE: This assumes BGR alignment!
	cv::cvtColor(rgb, imgBGR, CV_RGB2BGR);
	kfm.setKinectData(depth, imgBGR);

	// Results
	std::pair<DescSeg::Vec, DescTex::Vec> result;
	std::vector<int> keypointsSeg;
	std::vector<int> keypointsTex;
	keypointsSeg.clear();
	keypointsTex.clear();

	DescXYZ::Vec _temp = DescriptorUtil().fromPCL<pcl::PointXYZ, DescXYZ>(cloud, false);
	// Search structure in input
	typename CorrespondenceSearch<DescXYZ>::Ptr searchInput = CorrespondenceSearch<DescXYZ>::MakeXYZNNSearch(_temp, 1);


	if (!usePCD) kfm.computeExtendedLineSegments(_els, _ls, data3D2);
	else kfm.computeExtendedLineSegments(_els, _ls, data3D);


	string outXML;
	outXML.append("./xml/");
	outXML.append(xmlName);
	outXML.append("_ELS.xml");

	write3DExtendedLineSegmentsToXMLFile(*_els, outXML.c_str());


	//save temp images
	const_cast<SingleView<double>&>(kfm.getSingleView()).saveImages();
	// Store segments
	result.first = DescriptorEstimation().eseg<DescSeg>(*_els);
	keypointsSeg.reserve(result.first.size());
	for (DescSeg::Vec::const_iterator it = result.first.begin(); it != result.first.end(); ++it) {
		searchInput->query(*it);
		const int idx3D = searchInput->getNearestIdx().front();
		keypointsSeg.push_back(idx3D);
	}

	COVIS_ASSERT_MSG(!_ls->empty(), "No 2D line segments found!");
	COVIS_ASSERT_MSG(!_els->empty(), "No extended 3D line segments found!");


	//********************TEXLETS*******************************************************

	//texlets
	std::tr1::shared_ptr<std::vector<texlet2D> > texlet2Ds(new std::vector<texlet2D>);
	std::tr1::shared_ptr<std::vector<texlet3D> > texlet3Ds(new std::vector<texlet3D>);
	if (!usePCD) kfm.computeTexlets(texlet3Ds, texlet2Ds, data3D2, false);
	else kfm.computeTexlets(texlet3Ds, texlet2Ds, data3D, false);

	// Store texlets
	result.second = DescriptorEstimation().tex<DescTex>(*texlet3Ds);
	std::vector<texlet3D> texs = *texlet3Ds;
	// TODO: Keypoint: nearest input point
	keypointsTex.reserve(result.second.size());
	for (DescTex::Vec::const_iterator it = result.second.begin(); it != result.second.end(); ++it) {
		searchInput->query(*it);
		const int idx3D = searchInput->getNearestIdx().front();
		keypointsTex.push_back(idx3D);
	}

	// Check
	COVIS_ASSERT_MSG(!texlet2Ds->empty(), "No 2D texlets found!");
	COVIS_ASSERT_MSG(!texlet3Ds->empty(), "No 3D texlets found!");
	string outXMLtex;
	outXMLtex.append("./xml/");
	outXMLtex.append(xmlName);
	outXMLtex.append("_TEX.xml");
	write3DTexletsToXMLFile(*texlet3Ds, outXMLtex.c_str());

	//*************SURFACE*******************************************
	// TODO: Surface: uniform texlets with highest possible density
	Modules::KinectFeatureModule kfmSurf;
	kfmSurf.setCalibration(cc, w, h); // TODO
	kfmSurf.setFilterType(MONOGENIC_FREQ0110);
	kfmSurf.setTexletiDThresholds(0.01, 0.01);
	kfmSurf.setKinectData(depth, imgBGR);

	// Initialize
	COVIS_ASSERT(kfmSurf.Init());

	// Run
	std::tr1::shared_ptr<std::vector<texlet2D> > texlet2DSurf(new std::vector<texlet2D>);
	std::tr1::shared_ptr<std::vector<texlet3D> > texlet3DSurf(new std::vector<texlet3D>);

	if (!usePCD) kfmSurf.computeTexlets(texlet3DSurf, texlet2DSurf, data3D2, true);
	else kfmSurf.computeTexlets(texlet3DSurf, texlet2DSurf, data3D, true);

	DescECV::Vec surface = DescriptorEstimation().tex<DescECV>(*texlet3DSurf);

	bool _normalize = true;
	// Normalization
	if (_normalize) {
		normalize<DescECV>(surface);
		normalize<DescSeg>(result.first);
		normalize<DescTex>(result.second);
	} else {
		std::cerr << "Warning: normalization disabled during descriptor estimation!" << std::endl;
		std::cerr << "\tTo use the context descriptors for further processing such as correspondence search, remember to normalize first!" << std::endl;
	}

	//*********************************histogram



	return result;
																								}

