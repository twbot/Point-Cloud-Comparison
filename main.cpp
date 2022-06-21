#include <pcl/io/ply_io.h>
#include <pcl/io/io.h>
#include <pcl/io/file_io.h>
#include <pcl/io/ply/ply_parser.h>
#include <pcl/io/ply/ply.h>

#include <pcl/point_types.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

#include <pcl/common/transforms.h>
#include <pcl/common/geometry.h>
#include <pcl/common/common.h>
#include <pcl/common/common_headers.h>

#include <pcl/ModelCoefficients.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/gasd.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/range_image_border_extractor.h>

#include <pcl/filters/crop_box.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/keypoints/narf_keypoint.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>

#include <pcl/octree/octree_pointcloud_changedetector.h>

#include <pcl/registration/icp.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>

#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem>
#include <functional>
#include <regex>
#include <Eigen/Dense>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "./plyReader.h"

#define INAVLID_HEIGHT 55.555555
#define NOISE_REMOVAL_KDTREE 0
#define DEBUG 0

typedef std::vector<std::string> header;
typedef std::vector<std::array<float,3>> vertexData;
const std::string base_path = "DIRECTORY_WITH_POINT_CLOUD_DATA";

template<typename T>
bool load_ply(
	const std::string& input_filename,
	std::vector<T>& vertsCoordinates,
	std::vector<uchar>& vertsColours,
	header& comments
)
{
	try
	{
		pcl::console::print_highlight("Loading "); std::cout << input_filename <<  std::endl;

		std::ifstream ss_temp(input_filename, std::ios::binary);
		plyReader::PlyFile file_template(ss_temp);

		uint64_t vertexCount = file_template.request_properties_from_element("vertex", { "x", "y", "z" }, vertsCoordinates);
		uint64_t colourCount = file_template.request_properties_from_element("vertex", { "red", "green", "blue" }, vertsColours);

		comments = file_template.comments;

		if (vertexCount != (vertsCoordinates.size() / 3) && colourCount != (vertsColours.size() / 3))
		{
			std::cout << "Error: Only triangle mesh is supported. Abort" << std::endl;
			if (colourCount != (vertsColours.size() / 3)) {
				std::cout << "Error: Vertex colour count != vertex coordinate count. Abort" << std::endl;
			}
			return false;
		}

		file_template.read(ss_temp);

		std::cout << "Vertices : " << (!vertsCoordinates.empty() ? vertsCoordinates.size() / 3 : 0) << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: Could not load " << input_filename << ". " << e.what() << std::endl;
		return false;
	}
	return true;
}

// Downsample point cloud using voxel grid
void downSample(
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloudFiltered,
	float leafSize
)
{
	pcl::VoxelGrid<pcl::PointXYZRGB> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(leafSize, leafSize, leafSize); // leafSize was 0.85f
	sor.filter(*cloudFiltered);

	std::cout << "PointCloud before filtering: " << cloud->width * cloud->height
		<< " data points (" << pcl::getFieldsList(*cloud) << ")." << std::endl;
	std::cout << "PointCloud after filtering: " << cloudFiltered->width * cloudFiltered->height
		<< " data points (" << pcl::getFieldsList(*cloudFiltered) << ")." << std::endl;
}

void removeOutliers(
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr& inputCloud,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr& outCloud	
)
{
	std::cout << "Using statistical outlier removal..." << std::endl;
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
	sor.setInputCloud(inputCloud);
	sor.setMeanK(100);
	sor.setStddevMulThresh(1.0);
	sor.filter(*outCloud);
	inputCloud = outCloud;
}

void calculateNormals(
	pcl::PointCloud<pcl::PointXYZ>::Ptr& inputCloud,
	pcl::PointCloud<pcl::Normal>::Ptr& outputCloud
)
{
	std::cout << "Input dimension" << inputCloud->size() << std::endl;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZ>);
	kdTree->setInputCloud(inputCloud);

	// Normal Estimation
	std::cout << "Using normal method estimation...";
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> estimator;
	estimator.setNumberOfThreads(3);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	estimator.setInputCloud(inputCloud);
	estimator.setSearchMethod(kdTree);
	estimator.setKSearch(40); //It was 20
	estimator.compute(*normals);//Normals are estimated using standard method.

	// pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal> ());
	// pcl::concatenateFields(*inputCloud, *normals, *outputCloud);
	pcl::copyPointCloud(*normals, *outputCloud);

	std::cout << "Normal Estimation...[OK]" << std::endl;
}

cv::Mat generateHeightMap(
pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
float adjusted_width_x, float adjusted_width_y, float step,
float min_x, float min_y, float min_alt, float max_alt, int val
)
{
	std::cout << "Generating height map..." << std::endl;
	// Convert point to grid coordinate system
	const auto convert = [](
		float measure, float min, float step
	) {
		float adjustedMeasure = (measure - min) / step;
		return adjustedMeasure;
	};

	// Normalize elevation value between 0 and 255
	const auto convertAlt = [](
		float alt,
		float min_alt,
		float max_alt
	)
	{
		float val = (alt - min_alt) / (max_alt - min_alt);
		return val * 255;
	};

	float currAlt;
	cv::Mat matPlaneMask(cv::Size(adjusted_width_y, adjusted_width_x), CV_8UC1, 0.0);

	for (int i = 0; i < cloud->points.size(); i++) {
		currAlt = cloud->points[i].z;

		float row = convert(cloud->points[i].y, min_y, step);
		float col = convert(cloud->points[i].x, min_x, step);
		int alt = convertAlt(currAlt, min_alt, max_alt);
#if DEBUG
		if ( i < 100) {
			std::cout << "X: " << cloud->points[i].x << " Y: " << cloud->points[i].y << std::endl;
			std::cout << "Row: " << row << " col: " << col << std::endl;
			std::cout << "Curr Alt: " << currAlt << " Alt: " << alt << std::endl;
			std::cout << "====================" << std::endl;
		}
#endif
		if (row >= 0 && col >= 0 && row < adjusted_width_y && col < adjusted_width_x) {
			if (matPlaneMask.at<uchar>(row, col) < alt) {
				matPlaneMask.at<uchar>(row, col) = alt;
			}
		}
	}
	return matPlaneMask;
}

template <typename T>
void plyToPointCloud(
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
	std::vector<T>& vertexCoords,
	std::vector<uchar>& vertexColours,
	int val
) {
	pcl::PointXYZRGB pcl_point;
	for (int i = 0; i < vertexCoords.size(); i++) {
		const int index = i % 3;
		if (index == 0) {
			pcl_point.x = vertexCoords[i];
			pcl_point.r = vertexColours[i];
		}
		else if (index == 1) {
			pcl_point.y = vertexCoords[i];
			pcl_point.g = vertexColours[i];
		}
		else {
			pcl_point.z = vertexCoords[i];
			//if (i < 100 && val)
			//	std::cout << " Z: " << vertexCoords[i]  << " ZPCL: " << pcl_point.z << std::endl;
			pcl_point.b = vertexColours[i];
			cloud->points.push_back(pcl_point);
		}
	}
}

int main(
	int argc, 
	char** argv
) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ2(new pcl::PointCloud<pcl::PointXYZ>());

	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, 
		Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>> sourceClouds;
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr,
		Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>> keypointClouds;
	std::vector<pcl::PointCloud<pcl::Normal>::Ptr,
		Eigen::aligned_allocator<pcl::PointCloud<pcl::Normal>::Ptr>> normalClouds;
	std::array<header, 2> headers;
	std::array<vertexData, 2> pointData;
	std::array<cv::Mat, 2> heightmaps;
	std::array<std::vector<float>, 2> vertexCoordsInit;
	std::array<std::vector<uchar>, 2> vertexColoursInit;
	std::array<std::array<uint8_t, 3>, 2> colourKeyPoints;

	float min_x[2], min_y[2], max_x[2], max_y[2];
	float min_alt[2], max_alt[2], step[2];
	float adjusted_width_x[2], adjusted_width_y[2];

	// Seed timer
	srand(time(NULL));

	pcl::console::TicToc tt;

	const std::string ply1 = (argc > 1) ? argv[1] : base_path + "\\pointCloud.ply";
	const std::string ply2 = (argc > 2) ? argv[2] : base_path + "\\pointCloud2.ply";
	std::string select_leaf_size = (argc > 3) ? argv[3] : "0.95";
	float leaf_size = std::atof(select_leaf_size.c_str());
	colourKeyPoints[0][0] = 0; colourKeyPoints[0][1] = 255; colourKeyPoints[0][2] = 0;
	colourKeyPoints[1][0] = 0; colourKeyPoints[1][1] = 255; colourKeyPoints[1][2] = 255;

	if (!load_ply(ply1, vertexCoordsInit[0], vertexColoursInit[0], headers[0]) || 
		!load_ply(ply2, vertexCoordsInit[1], vertexColoursInit[1], headers[1]))
	{
		std::cout << "Error: Could not load ply file." << std::endl;
		return 1;
	}
	else {
		std::cout << "\nLoaded PLY Files[done, " << tt.toc() << " ms]" << std::endl;
	}

	const auto timeBeginAll = std::chrono::steady_clock::now();
	std::regex regexp("^Geometry, ([X-Z])=(-?[0-9]*\.[0-9]+):(-?[0-9]*\.[0-9]+):(-?[0-9]*\.[0-9]+)");
	std::smatch match;
	
	for (int i = 0; i < sizeof(headers) / sizeof(header); i++) {
		std::cout << "\nAdding data to point cloud " << i + 1 << std::endl;
		for (int j = 0; j < headers[i].size(); j++) {
			std::regex_search(headers[i][j], match, regexp);
			std::string coord = match.str(1);
			if (coord == "X") {
				min_x[i] = stof(match.str(2));
				max_x[i] = stof(match.str(4));
				step[i] = stof(match.str(3));
			}
			if (coord == "Y") {
				min_y[i] = stof(match.str(2));
				max_y[i] = stof(match.str(4));
			}
			if (coord == "Z") {
				min_alt[i] = stof(match.str(2));
				max_alt[i] = stof(match.str(4));
			}
		}
		adjusted_width_x[i] = (float)((max_x[i] - min_x[i]) / step[i]);
		adjusted_width_y[i] = (float)((max_y[i] - min_y[i]) / step[i]);

		// For debugging
#if DEBUG
		std::cout << "Min x: " << min_x[i] << " Max x: " << max_x[i] << std::endl;
		std::cout << "Min y: " << min_y[i] << " Max y: " << max_y[i] << std::endl;
		std::cout << "Min alt: " << min_alt[i] << " Max alt: " << max_alt[i] << std::endl;
		std::cout << "Step size: " << step[i] << std::endl;
		std::cout << "Adjusted_width_x: " << adjusted_width_x[i] << " Adjusted_width_y: " << adjusted_width_y[i] << std::endl;
#endif

		// Create temp point clouds for data storage
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

		// Convert vertices to points and fill respective point cloud
		plyToPointCloud(cloud, vertexCoordsInit[i], vertexColoursInit[i], i);
		vertexCoordsInit[i].clear(); vertexColoursInit[i].clear();

		cloud->width = (int)cloud->points.size();
		cloud->height = 1;
		cloud->is_dense = true;
		
		// Downsample point clouds
		downSample(cloud, cloudFiltered, leaf_size);
#if DEBUG
		for (int i = 0; i < 20; i++) {
			std::cout << "Cloud alt at " << i << " is " << cloud->points[i].z << std::endl;
		}
		std::cout << "=========================" << std::endl;
		for (int i = 0; i < 20; i++) {
			std::cout << "Cloud alt at " << i << " is " << cloudFiltered->points[i].z << std::endl;
		}
#endif
		// Filter outliers using point neighbourhood statistics
		//removeOutliers(cloudFiltered, cloudFiltered);

		// Start timer for height map generation + keypoint finding
		const auto timeB = std::chrono::steady_clock::now();

		// Generate heightmap from pointcloud
		heightmaps[i] = generateHeightMap(cloudFiltered, adjusted_width_x[i], adjusted_width_y[i],
			step[i], min_x[i], min_y[i], min_alt[i], max_alt[i], i);

		// AKAZE keypoint extractor
		std::cout << "Extracting keypoints from heightmap..." << std::endl;
		cv::Mat output;
		cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
		std::vector<cv::KeyPoint> keypoints;
		akaze->detectAndCompute(heightmaps[i], cv::noArray(), keypoints, output);
		std::cout << "Keypoint size: " << keypoints.size() << std::endl;

		// plot keypoints on img
		cv::Mat outputImg;
		cv::drawKeypoints(heightmaps[i], keypoints, outputImg, cv::Scalar(255.0, 0.0, 0.0, 0.0) );
		// For debugging
#if DEBUG
		cv::imwrite(base_path + "heightmap" + std::to_string(i) + ".jpg", heightmaps[i]);
		cv::imwrite(base_path + "keyPoints" + std::to_string(i) + ".jpg", outputImg);
#endif

		// Convert altitude to original units
		const auto convertAlt = [](
			float alt,
			float min_alt,
			float max_alt
			)
		{
			float val = (alt/255 * (max_alt - min_alt)) + min_alt;
			return val;
		};

		// Convert point to original coordinate system
		const auto convert = [](
			float measure, float min, float step
			) {
			float adjustedMeasure = (measure*step) + min;
			return adjustedMeasure;
		};

		for (std::size_t j = 0; j < keypoints.size(); j++) {
			pcl::PointXYZRGB point;
			float x_grid = keypoints[j].pt.x;
			point.x = convert(x_grid, min_x[i], step[i]);
			float y_grid = keypoints[j].pt.y;
			point.y = convert(y_grid, min_y[i], step[i]);
			uchar alt_norm = heightmaps[i].at<uchar>(keypoints[j].pt.y, keypoints[j].pt.x);
			point.z = convertAlt(alt_norm, min_alt[i], max_alt[i]);
			point.r = colourKeyPoints[i][0];
			point.g = colourKeyPoints[i][1];
			point.b = colourKeyPoints[i][2];
			keypointCloud->points.push_back(point);
		}
		keypointClouds.push_back(keypointCloud);
		sourceClouds.push_back(cloudFiltered);
		const auto timeS = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::seconds> (timeS - timeB);
		std::cout << "Time Duration Heightmap: " << duration.count() << " seconds" << std::endl;
		cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
		keypointCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloudFiltered.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
	}

	// Point Cloud Registration
	// Start timer
	const auto timeBegin = std::chrono::steady_clock::now();
	std::cout << "\nRegistering point clouds..." << std::endl;
	pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
	icp.setInputSource(keypointClouds[1]);
	icp.setInputTarget(keypointClouds[0]);
	icp.setTransformationEpsilon(1e-9);
	icp.align(*keypointClouds[1]);

	Eigen::Matrix4f transformationMatrix;

	if (icp.hasConverged())
	{
		std::cout << "ICP converged." << std::endl
			<< "The score is " << icp.getFitnessScore() << std::endl;
		transformationMatrix = icp.getFinalTransformation();
	}
	else {
		std::cout << "ICP did not converge." << std::endl;
		transformationMatrix = icp.getFinalTransformation();
	}
	std::cout << "Transformation matrix: " << std::endl;
	std::cout << transformationMatrix << std::endl;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr tempCloudTransform(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud(*sourceClouds[1], *tempCloudTransform);
	pcl::transformPointCloud(*tempCloudTransform, *sourceClouds[1], transformationMatrix);

	// End timer
	const auto timeStop = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (timeStop - timeBegin);
	std::cout << "Time Duration ICP: " << duration.count() << " milliseconds" << std::endl;

	// Octree resolution - side length of octree voxels
	float resolution = 5.0f;

	// Instantiate octree-based point cloud change detection class
	pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZRGB> octree(resolution);
	octree.setInputCloud(sourceClouds[1]);
	octree.addPointsFromInputCloud();

	octree.switchBuffers();

	// Add points from cloudB to octree 
	octree.setInputCloud(sourceClouds[0]);
	octree.addPointsFromInputCloud();

	std::vector<int> newPointIdxVector;

	// Get vector of point indices from octree voxels which did not exist in previous buffer
	octree.getPointIndicesFromNewVoxels(newPointIdxVector, 7);

	pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZRGB> octreeAdditions(resolution);
	octreeAdditions.setInputCloud(sourceClouds[0]);
	octreeAdditions.addPointsFromInputCloud();

	octreeAdditions.switchBuffers();

	// Add points from cloudB to octree 
	octreeAdditions.setInputCloud(sourceClouds[1]);
	octreeAdditions.addPointsFromInputCloud();

	std::vector<int> newPointIdxVectorAdditions;

	// Get vector of point indices from octree voxels which did not exist in previous buffer
	octreeAdditions.getPointIndicesFromNewVoxels(newPointIdxVectorAdditions, 7);

	// filter out sparse differenced voxels
	/*
	TODO: For each differenced voxel, find adjacent octree voxels within a given radius R and if <D of adjacent
			voxels are differenced voxels, delete differenced voxel
	*/

	std::cout << "Adding point addition data ... " << "              ";
	for (std::size_t i = 0; i < newPointIdxVectorAdditions.size(); ++i) {
		pcl::PointXYZRGB point = (*sourceClouds[1])[newPointIdxVectorAdditions[i]];
		point.r = 0;
		point.g = 255;
		point.b = 0; 
		sourceClouds[1]->points.push_back(point);
	}
	std::cout << "Point Addition Size: " << newPointIdxVectorAdditions.size() << std::endl;

	std::cout << "Adding point subtraction data ... " << "           ";
	for (std::size_t i = 0; i < newPointIdxVector.size(); ++i) {
		pcl::PointXYZRGB point = (*sourceClouds[0])[newPointIdxVector[i]];
		point.r = 255;
		point.g = 0;
		point.b = 0;
		sourceClouds[1]->points.push_back(point);
	}
	std::cout << "Point Subtraction Size: " << newPointIdxVector.size() << std::endl;

	const auto timeStopAll = std::chrono::steady_clock::now();
	auto durationAll = std::chrono::duration_cast<std::chrono::seconds> (timeStopAll - timeBeginAll);
	std::cout << "Time Duration All: " << durationAll.count() << " seconds" << std::endl;

#if NOISE_REMOVAL_KDTREE
	std::cout << "Removing difference noise..." << std::endl;
	std::cout << "Size of newPointIdxVector: " << newPointIdxVector.size() << std::endl;
	for (std::size_t i = 0; i < newPointIdxVector.size(); i++) {
		pcl::PointXYZRGB searchPoint = (*cloud2Transformed)[newPointIdxVector[i]];

		// Number of points to be considered non-noise
		int kValue = 15;
		int numNeighbours = 0;
		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquareDistance;

		//kdTree init
		pcl::KdTreeFLANN<pcl::PointXYZRGB> kdTree;
		kdTree.setInputCloud(cloud2Transformed);
		kdTree.setSortedResults(true);

		if (kdTree.radiusSearch(searchPoint, 30, pointIdxRadiusSearch, pointRadiusSquareDistance) > 0)
		{
			for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++) {
				pcl::PointXYZRGB pointAt = (*cloud2Transformed)[pointIdxRadiusSearch[j]];
				if (pointAt.g == 0 && pointAt.r == 255 && pointAt.b == 0) {
					numNeighbours++;
				}
			}
		}
		if (!(numNeighbours >= kValue)) {
			std::cout << "Found " << numNeighbours;
			cloud2Transformed->points.erase(cloud2Transformed->points.begin() + newPointIdxVector[i]);
			std::cout << " neighbours" << std::endl;
			//cloud2Transformed->points.erase(cloud2Transformed->points.begin() + newPointIdxVector[i]);
		}
	}

	// Remove all differencing points with number of neighbours less than k
	std::cout << "Removing differencing noise..." << std::endl;
	std::cout << "Size of newPointIdxVectorAdditions: " << newPointIdxVectorAdditions.size() << std::endl;
	for (std::size_t i = 0; i < newPointIdxVectorAdditions.size(); i++) {
		pcl::PointXYZRGB searchPoint = (*cloud2Transformed)[newPointIdxVectorAdditions[i]];

		// Number of points to be considered non-noise
		int kValue = 15;
		int numNeighbours = 0;
		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquareDistance;

		// kdTree init
		pcl::KdTreeFLANN<pcl::PointXYZRGB> kdTree;
		kdTree.setInputCloud(cloud2Transformed);
		kdTree.setSortedResults(true);

		if (kdTree.radiusSearch(searchPoint, 30, pointIdxRadiusSearch, pointRadiusSquareDistance) > 0)
		{
			for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++) {
				pcl::PointXYZRGB pointAt = (*cloud2Transformed)[pointIdxRadiusSearch[j]];
				if (pointAt.g == 255 && pointAt.r == 0 && pointAt.b == 0) {
					numNeighbours++;
				}
			}
		}

		if (!(numNeighbours >= kValue)) {
			std::cout << "Found " << numNeighbours;
			cloud2Transformed->points.erase(cloud2Transformed->points.begin() + newPointIdxVectorAdditions[i]);
			std::cout << " neighbours" << std::endl;
			//cloud2Transformed->points.erase(cloud2Transformed->points.begin() + newPointIdxVector[i]);
		}
		
	}
#endif

	boost::shared_ptr<pcl::visualization::PCLVisualizer> view(new pcl::visualization::PCLVisualizer);
	view->addPointCloud(sourceClouds[1]);
	while (!view->wasStopped())
	{
		view->spinOnce(100);
	}

	system("pause");

	return (0);
}
