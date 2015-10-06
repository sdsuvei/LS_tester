#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "flann/flann.hpp"

#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/make_shared.hpp>
#include <pcl/point_representation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

#include <iostream>
#include <stdio.h>

using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//using namespace cv;
using namespace std;

int block_estimation = 0;
int min_estimation = 0;
int max_estimation = 0;
int puncte = 0;
cv::Mat disp, disp8;
pcl::visualization::PCLVisualizer *p;

//convenient typedefs
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;


static void print_help()
{
	printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
	printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|var] [--blocksize=<block_size>]\n"
			"[--max-disparity=<max_disparity>] --min-disparity=<min_disparity>] [--scale=scale_factor>] [-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"
			"[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");
}

vector<cv::Point2f> nn( vector<cv::Point2f> src, cv::Mat edges) {
	 vector<cv::Point2f> dst;

    for(size_t i = 0; i < src.size(); ++i) {
        // Edge point coordinate, given in template
        const int c = src[i].x;
        const int r = src[i].y;
        // Global image coordinates of edge point
        const int cglobal = c;
        const int rglobal = r;
        // Search for nearest destination point around the center point (rglobal,cglobal)
        int rad = 1; // Search boxes of increasing size
        // Distance and coordinate of closest matching edge point
        int mindist = INT_MAX;
        int rmin = -1;
        int cmin = -1;
        // Search increasingly big boxes until we find a match or until we reach the image border
        while(mindist == INT_MAX && rad<21) {
            // Search the current box
            for(int rr = max(0, rglobal-rad); rr <= min(edges.rows-1, rglobal+rad); ++rr) {
                for(int cc = max(0, cglobal-rad); cc <= min(edges.cols-1, cglobal+rad); ++cc) {
                    if(edges.at<int16_t>(rr,cc) > 0) { // If current point is non-zero
                        const int dist = abs(rglobal-rr) + abs(cglobal-cc);
                        if(dist < mindist) {
                            mindist = dist;
                            rmin = rr;
                            cmin = cc;
                        }
                    }
                }
            }
            // Expand the box
            ++rad;
        }

        // Save corresponding point
        //dst.x=cmin;
        //dst.y=rmin;
        dst.push_back(cv::Point2f(cmin,rmin));
    }

    return dst;
}

static void showXYZ(const cv::Mat& mat, const cv::Mat& rgb, const string& outfile="")
{
	pcl::PointCloud<pcl::PointXYZRGB> cloud;
	cloud.reserve(mat.cols*mat.rows);

	const double max_z = 2e3; // Disregard points farther away than 2 m
	//const double max_z = 1.24e3; // Disregard points farther away than 2 m
	//FILE* fp = fopen(filename, "wt");
	for(int y = 0; y < mat.rows; y++)
	{
		for(int x = 0; x < mat.cols; x++)
		{
			cv::Vec3f point = mat.at<cv::Vec3f>(y, x);

			// This omits zero points
			if(point[0] == 0 && point[1] == 0 && point[2] == 0)
				continue;

			// This omits points equal to or larger than max_z
			if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)
				continue;
			//fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);

			// Point to write to
			pcl::PointXYZRGB p;

			// Scale position from mm to m
			p.x = 0.001*point[0];
			p.y = 0.001*point[1];
			p.z = 0.001*point[2];

			// OpenCV reads in images in BGR order, so we must switch to BGR for PCL
			cv::Vec3b pbgr = rgb.at<cv::Vec3b>(y,x);
			p.b = pbgr[0];
			p.g = pbgr[1];
			p.r = pbgr[2];

			cloud.push_back(p);
		}
	}

	cout << "Showing " << cloud.size() << " points" << endl;
	// Show using PCL
	pcl::visualization::PCLVisualizer visu;
	visu.addPointCloud(cloud.makeShared()); //it's ok with error
	visu.addCoordinateSystem(0.25);
	visu.setBackgroundColor(0.5,0.5,0.5);
	visu.spin();

	if(!outfile.empty())
		pcl::io::savePCDFile(outfile, cloud);
}


void showCloudsLeft(PointCloud::Ptr cloud_target, PointCloud::Ptr cloud_source)
{
  p->removePointCloud ("vp1_target");
  p->removePointCloud ("vp1_source");

  PointCloudColorHandlerCustom<PointT> tgt_h (cloud_target, 0, 255, 0);
  PointCloudColorHandlerCustom<PointT> src_h (cloud_source, 255, 0, 0);
  p->addPointCloud (cloud_target, tgt_h, "vp1_target");
  p->addPointCloud (cloud_source, src_h, "vp1_source");
  p-> spin();
}


int main (int argc, char** argv)
{

	const char* algorithm_opt = "--algorithm=";
		const char* maxdisp_opt = "--max-disparity=";
		const char* mindisp_opt = "--min-disparity=";
		const char* blocksize_opt = "--blocksize=";
		const char* nodisplay_opt = "--no-display=";
		const char* scale_opt = "--scale=";

		if(argc < 3) {
			print_help();
			return 0;
		}
		const char* img1_filename = 0; //left image
		const char* img2_filename = 0; //right image
		const char* intrinsic_filename = 0; //intrinsic parameters
		const char* extrinsic_filename = 0; //extrinsic parameters
		const char* disparity_filename = 0;
		const char* point_cloud_filename = 0;

		int iterator=0;

		 // Create a window
		cv::namedWindow("Disparity", cv::WINDOW_NORMAL);
		cv::resizeWindow("Disparity", 900,900);

		 //Create trackbar to change block size
		int block_slider = 5;
		cv::createTrackbar("Block size", "Disparity", &block_slider, 127);

		 //Create trackbar to change minimum disparity
		int min_slider = 30;
		cv::createTrackbar("Min disparity", "Disparity", &min_slider, 1000);

		 //Create trackbar to change maximum disparity
		int max_slider = 120;
		cv::createTrackbar("Max disparity", "Disparity", &max_slider, 1000);

		enum { STEREO_BM=0 };
		int alg = STEREO_BM;
		int SADWindowSize = 0, numberOfDisparities = 0, min_Disparities = 0;
		bool no_display = false;
		float scale = 1.f;

		cv::StereoBM bm;

		//read commandline options
		for( int i = 1; i < argc; i++ )
		{
			if( argv[i][0] != '-' )
			{
				if( !img1_filename )
					img1_filename = argv[i];
				else
					img2_filename = argv[i];
			}
			else if( strncmp(argv[i], maxdisp_opt, strlen(maxdisp_opt)) == 0 )
			{
				if( sscanf( argv[i] + strlen(maxdisp_opt), "%d", &numberOfDisparities ) != 1 ||
						numberOfDisparities < 1 || numberOfDisparities % 16 != 0 )
				{
					printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
					print_help();
					return -1;
				}
			}
			else if( strncmp(argv[i], mindisp_opt, strlen(mindisp_opt)) == 0 )
			{
				if( sscanf( argv[i] + strlen(mindisp_opt), "%d", &min_Disparities ) != 1 )
				{
					printf("Command-line parameter error: The min disparity\n");
					print_help();
					return -1;
				}
			}
			else if( strncmp(argv[i], blocksize_opt, strlen(blocksize_opt)) == 0 )
			{
				if( sscanf( argv[i] + strlen(blocksize_opt), "%d", &SADWindowSize ) != 1 ||
						SADWindowSize < 1 || SADWindowSize % 2 != 1 )
				{
					printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
					return -1;
				}
			}
			else if( strncmp(argv[i], scale_opt, strlen(scale_opt)) == 0 )
			{
				if( sscanf( argv[i] + strlen(scale_opt), "%f", &scale ) != 1 || scale < 0 )
				{
					printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
					return -1;
				}
			}
			else if( strcmp(argv[i], nodisplay_opt) == 0 )
				no_display = true;
			else if( strcmp(argv[i], "-i" ) == 0 )
				intrinsic_filename = argv[++i];
			else if( strcmp(argv[i], "-e" ) == 0 )
				extrinsic_filename = argv[++i];
			else if( strcmp(argv[i], "-o" ) == 0 )
				disparity_filename = argv[++i];
			else if( strcmp(argv[i], "-p" ) == 0 )
				point_cloud_filename = argv[++i];
			else
			{
				printf("Command-line parameter error: unknown option %s\n", argv[i]);
				return -1;
			}
		}

		if( !img1_filename || !img2_filename )
		{
			printf("Command-line parameter error: both left and right images must be specified\n");
			return -1;
		}

		if( (intrinsic_filename != 0) ^ (extrinsic_filename != 0) )
		{
			printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
			return -1;
		}

		if( extrinsic_filename == 0 && point_cloud_filename )
		{
			printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
			return -1;
		}

		int color_mode = alg == STEREO_BM ? 0 : -1;
		cv::Mat img1 = cv::imread(img1_filename, color_mode);
		cv::Mat img2 = cv::imread(img2_filename, color_mode);

		cv::Mat img_colored = cv::imread(img1_filename, -1);

		if( scale != 1.f ) {
			cv::Mat temp1, temp2;
			int method = scale < 1 ? cv::INTER_AREA : cv::INTER_CUBIC;
			cv::resize(img1, temp1, cv::Size(), scale, scale, method);
			img1 = temp1;
			cv::resize(img2, temp2, cv::Size(), scale, scale, method);
			img2 = temp2;
		}

		cv::Size img_size = img1.size();

		cv::Rect roi1, roi2;
		cv::Mat Q;

		cv::Mat R, T, R1, P1, R2, P2;

		if( intrinsic_filename )
		{
			// reading intrinsic parameters
			cv::FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
			if(!fs.isOpened())
			{
				printf("Failed to open file %s\n", intrinsic_filename);
				return -1;
			}

			cv::Mat M1, D1, M2, D2;
			fs["M1"] >> M1;
			fs["D1"] >> D1;
			fs["M2"] >> M2;
			fs["D2"] >> D2;

			M1 *= scale;
			M2 *= scale;

			fs.open(extrinsic_filename, CV_STORAGE_READ);
			if(!fs.isOpened())
			{
				printf("Failed to open file %s\n", extrinsic_filename);
				return -1;
			}

			//Mat R, T, R1, P1, R2, P2;
			fs["R"] >> R;
			fs["T"] >> T;

			//stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );
			cv::stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, 0, 0, img_size, &roi1, &roi2 );

			if(iterator<1)
			{
				std::cout<<"R1 matrix:\n "<<R1<<endl<<endl;
				std::cout<<"R2 matrix:\n "<<R2<<endl<<endl;
				std::cout<<"P1 matrix:\n "<<P1<<endl<<endl;
				std::cout<<"P2 matrix:\n "<<P2<<endl<<endl;

				std::cout<<"P1 (0 3): "<<P1.at<double>(0,2)<<endl;
			}

			cv::Mat map11, map12, map21, map22;
			cv::initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
			cv::initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

			cv::Mat img1r, img2r, img_cr;

			cv::remap(img1, img1r, map11, map12, cv::INTER_CUBIC);
			cv::remap(img2, img2r, map21, map22, cv::INTER_CUBIC);

			cv::remap(img_colored, img_cr, map11, map12, cv::INTER_CUBIC);

			img1 = img1r;
			img2 = img2r;
			img_colored = img_cr;

			if(iterator<1)
			{
				cv::imwrite("left_rect.png",img1);
				cv::imwrite("right_rect.png",img2);
			}

		}

		// Start of trackbar functionality
		while(true)
		{

			numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;
			if(iterator<1)
			{
				bm.state->roi1 = roi1;
				bm.state->roi2 = roi2;
				bm.state->preFilterCap = 31;
				bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
				bm.state->minDisparity = min_Disparities;
				bm.state->numberOfDisparities = numberOfDisparities;
				bm.state->textureThreshold = 10;
				bm.state->uniquenessRatio = 15;
				bm.state->speckleWindowSize = 100;
				bm.state->speckleRange = 32;
				bm.state->disp12MaxDiff = 1;
			}
			else
			{
				block_estimation=cv::getTrackbarPos("Block size", "Disparity")*2+1;
				if(block_estimation<5)
					block_estimation=5;
				min_estimation = cv::getTrackbarPos("Min disparity", "Disparity");
				max_estimation = cv::getTrackbarPos("Max disparity", "Disparity");
				if(max_estimation<16)
					max_estimation=16;
				else
					max_estimation=max_estimation-(max_estimation%16);
				bm.state->roi1 = roi1;
				bm.state->roi2 = roi2;
				bm.state->preFilterCap = 31;
				bm.state->SADWindowSize = block_estimation;
				bm.state->minDisparity = min_estimation;
				bm.state->numberOfDisparities = max_estimation;
				bm.state->textureThreshold = 10;
				bm.state->uniquenessRatio = 15;
				bm.state->speckleWindowSize = 100;
				bm.state->speckleRange = 32;
				bm.state->disp12MaxDiff = 1;
			}

			int cn = img1.channels();

			int64 t = cv::getTickCount();
			bm(img1, img2, disp);
			t = cv::getTickCount() - t;
			//printf("Time elapsed for stereo matching: %fms\n", t*1000/getTickFrequency());
			disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));

			//Show result
			cv::imshow("Disparity", disp);

			iterator+=1;

			// Wait until user press some key for 50ms
			int iKey = cv::waitKey(50);

			// If user press 'ESC' key
			if (iKey == 27)
				{
					cout<<"Program finished!"<<endl;
					printf("Time elapsed for stereo matching: %fms\n", t*1000/cv::getTickFrequency());
					cout<<"The best found parameters are:"<<endl;
					cout<<"-> Block size = "<<block_estimation<<endl;
					cout<<"-> Minimum disparity = "<<min_estimation<<endl;
					cout<<"-> Maximum disparity = "<<max_estimation<<endl;
					break;
				}
			}

		if(disparity_filename)
			cv::imwrite(disparity_filename, disp8);

		if(point_cloud_filename) {
			printf("Storing the point cloud...");
			fflush(stdout);

			if(iterator<1)
			{
				std::cout<<"Q reproject matrix:\n "<<Q<<endl;
				std::cout<<"Q (0 3): "<<Q.at<double>(0,3)<<endl;
			}

			// Get parameters for reconstruction
			float f = P1.at<double>(0,0); // Focal length
			float B = P2.at<double>(0,3)/f; // Baseline in the x direction
			float cx = P1.at<double>(0,2); // Center x coordinate
			float cy = P1.at<double>(1,2); // Center y coordinate

			float cx2 = P2.at<double>(0,2); // Center x coordinate of right image
			float dcx = cx-cx2; // Difference in center x coordinates
			int temp = disp.at<int16_t>(0,0);
			int maxdisp = 0;
			for(int y = 0; y < disp.rows; ++y) {
				for(int x = 0; x<disp.cols; ++x) {
					if(temp > disp.at<int16_t>(y,x))
						temp = disp.at<int16_t>(y,x);
					if(maxdisp < disp.at<int16_t>(y,x))
						maxdisp = disp.at<int16_t>(y,x);
				}
			}

			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr out (new pcl::PointCloud<pcl::PointXYZRGBA>());
			out->height = disp.cols;
			out->width = disp.rows;
			out->points.resize(out->height * out->width);


			for (int i = 0; i < out->size(); i++){
				(*out)[i].x = std::numeric_limits<float>::quiet_NaN();
				(*out)[i].y = std::numeric_limits<float>::quiet_NaN();
				(*out)[i].z = std::numeric_limits<float>::quiet_NaN();
			}

			cv::Mat_<cv::Vec3f> xyz(disp.rows, disp.cols, cv::Vec3f(0,0,0)); // Resulting point cloud, initialized to zero
			for(int y = 0; y < disp.rows; ++y) {
				for(int x = 0; x < disp.cols; ++x) {
					pcl::PointXYZRGBA point;

					// Avoid invalid disparities
					if(disp.at<int16_t>(y,x) == temp) continue;
					if(disp.at<int16_t>(y,x) == 0) continue;

					float d = float(disp.at<int16_t>(y,x)) / 16.0f; // Disparity
					float W = B/(-d+dcx); // Weighting

					point.x = (float(x)-cx) * W;
					point.y = (float(y)-cy) * W;
					point.z = f * W;
					//skip 0 points
					if (point.x== 0 && point.y == 0 && point.z == 0) continue;
					// disregard points farther then 2m
					const double max_z = 2e3;
					if (fabs(point.y - max_z) < FLT_EPSILON || fabs(point.y) > max_z) continue;
					//scale position from mm to m
					point.x = 0.001*point.x;
					point.y = 0.001*point.y;
					point.z = 0.001*point.z;
					//add color
					cv::Vec3b bgr = img_colored.at<cv::Vec3b>(y,x);
					point.b = bgr[0];
					point.g = bgr[1];
					point.r = bgr[2];

					out->at(y, x) = point;
				}
			}
			pcl::io::savePCDFile("stereo.pcd", *out);

			//saveXYZ(point_cloud_filename, xyz);
			//showXYZ(xyz, img_colored, point_cloud_filename);
			printf("\n");
		}



  pcl::PointCloud<pcl::PointXYZ>::Ptr stereo_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud (new pcl::PointCloud<pcl::PointXYZ>);

  // Loading the two point clouds
  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("stereo.pcd", *stereo_cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read the STEREO point cloud. \n");
    return (-1);
  }

  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("lidar.pcd", *lidar_cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read the LIDAR point cloud. \n");
    return (-1);
  }

  //The Transform Matrix which moves the lidar points in the stereo camera frame
  Eigen::Matrix4f transform_lTs = Eigen::Matrix4f::Identity();
  transform_lTs(0,0) = -0.0282604;
  transform_lTs(0,1) = 0.997477;
  transform_lTs(0,2) =	0.0656693;
  transform_lTs(0,3) = 0.815621;
  transform_lTs(1,0) = 0.136367;
  transform_lTs(1,1) = 0.0689264;
  transform_lTs(1,2) = -0.988274;
  transform_lTs(1,3) = 0.049019;
  transform_lTs(2,0) = -0.990292;
  transform_lTs(2,1) = -0.0189803;
  transform_lTs(2,2) = -0.137977;
  transform_lTs(2,3) = 2.12426;
  transform_lTs(3,0) = 0;
  transform_lTs(3,1) = 0;
  transform_lTs(3,2) = 0;
  transform_lTs(3,3) = 1;


  // Executing the transformation
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  // You can either apply transform_1 or transform_2; they are the same
  pcl::transformPointCloud (*lidar_cloud, *transformed_cloud, transform_lTs);

  // Visualization
  /* printf(  "\nPoint cloud colors :  white  = original point cloud\n"
       "                        red  = transformed point cloud\n");
   pcl::visualization::PCLVisualizer viewer ("Transformation of the Lidar point cloud");

   // Define R,G,B colors for the point cloud
   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler (lidar_cloud, 255, 255, 255);
   // We add the point cloud to the viewer and pass the color handler
   viewer.addPointCloud (lidar_cloud, source_cloud_color_handler, "original_cloud");

   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler (transformed_cloud, 230, 20, 20); // Red
   viewer.addPointCloud (transformed_cloud, transformed_cloud_color_handler, "transformed_cloud");
   viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
   viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
   viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");
   //viewer.setPosition(800, 400); // Setting visualiser window position

   while (!viewer.wasStopped ()) { // Display the visualiser until 'q' key is pressed
     viewer.spinOnce ();
   }*/

   // Visualize the new transformed Lidar point cloud over the stereo one
  // p = new pcl::visualization::PCLVisualizer (argc, argv, "Pairwise Incremental Registration example");
  // showCloudsLeft(transformed_cloud, stereo_cloud);

   pcl::PointCloud<pcl::PointXYZ> Lcloud=*transformed_cloud;
   pcl::PointCloud<pcl::PointXYZ> saved_cloud;
   saved_cloud.height=Lcloud.height;
   saved_cloud.width=Lcloud.width;

   for(int s=0;s<saved_cloud.points.size(); ++s)
   {
	   saved_cloud.points[s].x=0;
	   saved_cloud.points[s].y=0;
	   saved_cloud.points[s].z=0;
   }


   cv::Mat lidar_l=cv::Mat::zeros(img1.rows, img1.cols, CV_8U); // create empty disparity map for Lidar data

   float f = P1.at<double>(0,0); // Focal length
   float B = P2.at<double>(0,3)/f; // Baseline in the x direction
   float cx = P1.at<double>(0,2); // Center x coordinate
   float cy = P1.at<double>(1,2); // Center y coordinate

   float cx2 = P2.at<double>(0,2); // Center x coordinate of right image
   float cy2 = P2.at<double>(1,2); // Center y coordinate of right image
   float dcx = cx-cx2;

   for (int i = 0; i < Lcloud.points.size (); ++i)  //try with cloud height and width
   {
	   // cross reference the 2D points
	   double x_=806.6065420854103*Lcloud.points[i].x+982.1147003173828*Lcloud.points[i].z;
	   double y_=806.6065420854103*Lcloud.points[i].y+515.74658203125*Lcloud.points[i].z;
	   double z_=Lcloud.points[i].z;

	   int x_2d=x_/z_;
	   int y_2d=y_/z_;

	   //std::cout<<"x_2d: "<<x_2d<<" y_2d: "<<y_2d<<endl;

	   //scale position from m to mm;
	   Lcloud.points[i].x = Lcloud.points[i].x/0.001;
	   Lcloud.points[i].y = Lcloud.points[i].y/0.001;
	   Lcloud.points[i].z = Lcloud.points[i].z/0.001;

	   float d = (Lcloud.points[i].z*dcx-f*B)/Lcloud.points[i].z; // disparity
	   float W = B/(-d+dcx); // Weighting
	   int x = (Lcloud.points[i].x+cx*W)/W; // x value
	   int y = (Lcloud.points[i].y+cy*W)/W; // y value
	   //std::cout<<"x: "<<x<<" y: "<<y<<endl;
	   if(y>=0 && y<lidar_l.rows && x>=0 && x<lidar_l.cols)
	   {
		   lidar_l.at<int16_t>(y,x)=d;
		   //std::cout<<"x: "<<x<<" y: "<<y<<" d: "<<d<<endl;
		   saved_cloud.points.push_back(Lcloud.points[i]);
	   }
   }

   // KdTree with 5 random trees
   	cv::flann::KDTreeIndexParams indexParams(1);
   	//Mat lidar_points = cv::Mat(img1.size(), 2,CV_32F); // --- lidar points
   	//Mat stereo_points = cv::Mat(lidar_l.size(), 2,CV_32F); // --- image points

   	cv::Mat lidar_points = cv::Mat::zeros(12717, 2, CV_32F);
   	cv::Mat stereo_points = cv::Mat::zeros(20736000, 2, CV_32F);

    int c_l = 0;
    int c_s = 0;

   	for(int w = 0; w < lidar_l.rows; ++w) {
 		for(int v = 0; v < lidar_l.cols; ++v) {
 			if(lidar_l.at<int16_t>(w,v)!=0)
 				{
 				lidar_points.at<float>(c_l,0)=w;
 				lidar_points.at<float>(c_l,1)=v;
 				c_l+=1;
 				}

 		}
 	}

   	for(int w = 0; w < img1.rows; ++w) {
   	 		for(int v = 0; v < img1.cols; ++v) {
   	 		if(img1.at<int16_t>(w,v)!=0) //STEREO.at<int16_t>(w,v)!=0
   	 			{
 				stereo_points.at<float>(c_s,0)=w;
 				stereo_points.at<float>(c_s,1)=v;
 				c_s+=1;
 				}
   	 		}
   	 	}

   	std::cout<<"c_l: "<<c_l<<" c_s: "<<c_s<<endl;
	
    /*
     * AGB EXAMPLE CODE USING FLANN BEGIN
     */
    {
	// Convert your 2D lidar points to FLANN matrix
	flann::Matrix<float> lidar_points_flann(lidar_points.ptr<float>(), lidar_points.rows, lidar_points.cols);
	
	// Create single k-d tree
	flann::KDTreeSingleIndex<flann::L2<float> > kdtree_flann(lidar_points_flann);
	kdtree_flann.buildIndex();
	
	// Convert the 2D stereo points to FLANN
	flann::Matrix<float> stereo_points_flann(stereo_points.ptr<float>(), stereo_points.rows, stereo_points.cols);
	
	// Do search
	vector<vector<size_t> > indices_flann;
	vector<vector<float> > dists_flann;
	kdtree_flann.knnSearch(stereo_points_flann, indices_flann, dists_flann, 1, flann::SearchParams());
	
	// INFO:
	// Now each outer index of indices_flann/dists_flann corresponds to each row of stereo_points
	// Each inner vector of indices_flann/dists_flann is just a vector of 1 element because we are just doing a 1-NN search
    }
    /*
     * AGB EXAMPLE CODE USING FLANN END
     */

    // Create the Index
    cv::flann::Index kdtree(lidar_points, indexParams);
    cv::Mat indices, dists = cv::Mat::zeros(c_s, 1,CV_32F);

	int64 t2 = cv::getTickCount();
    kdtree.knnSearch(stereo_points, indices, dists, 1, cv::flann::SearchParams(8));
	t2 = cv::getTickCount() - t2;
	printf("Time elapsed for FLANN: %fms\n", t2*1000/cv::getTickFrequency());


    flann::Matrix<float> query;

     cout << "Output::"<< endl;
	for(int row = 0 ; row < indices.rows ; row++){
		cout << "(index,dist):";
		for(int col = 0 ; col < indices.cols ; col++){
			cout << "(" << indices.at<int>(row,col) << "," << dists.at<float>(row,col) << ")" << "\t";
		}
		cout << endl;
	}
	/*
vector<Point2f> stereo;
   for(int w = 0; w < img1.rows; ++w) {
		for(int v = 0; v < img1.cols; ++v) {
			Point2f fp( (float)(w),(float)(v));
	        stereo.push_back(fp);
		}
	}
	std::cout<<"Computing NN ... "<<endl;
	int64 t1 = getTickCount();
	vector<Point2f> lidar_nn=nn(stereo,lidar_l);
	t1 = getTickCount() - t1;
	printf("Time elapsed for NN: %fms\n", t1*1000/getTickFrequency());

	for(int s = 0; s < lidar_nn.size(); ++s) {
			std::cout<<lidar_nn[s]<<endl;
	}*/

    saved_cloud.width = 1;
    saved_cloud.height = saved_cloud.points.size();

   	pcl::io::savePCDFileASCII ("saved_lidar.pcd", saved_cloud);

	//Show result of Lidar->image
	cv::imwrite("lidar_disp.png",lidar_l);
    //namedWindow("Lidar_image", WINDOW_AUTOSIZE );
	//imshow("Lidar_image", lidar_l);
	//waitKey(0);
	std::cout<<"The initial LIDAR point cloud has "<<Lcloud.points.size()<<" points "<<endl;
	std::cout<<"The new LIDAR point cloud has "<<saved_cloud.points.size()<<" points "<<endl;
	std::cout<<"Program done "<<endl;

  return (0);
}
