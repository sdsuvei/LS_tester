#include "lidar_stereo.h"

int main (int argc, char** argv)
{

	//const char* algorithm_opt = "--algorithm=";
	//const char* maxdisp_opt = "--max-disparity=";
	//const char* mindisp_opt = "--min-disparity=";
	//const char* blocksize_opt = "--blocksize=";
	//const char* nodisplay_opt = "--no-display=";
	//const char* scale_opt = "--scale=";

	if(argc < 3) {
		print_help();
		return 0;
	}

	const char* img1_filename = 0; //left image
	const char* img2_filename = 0; //right image
	const char* intrinsic_filename = 0; //intrinsic parameters
	const char* extrinsic_filename = 0; //extrinsic parameters
	const char* disparity_filename1 = 0;
	const char* disparity_filename2 = 0;
	const char* experiment_filename_1 = 0;
	const char* experiment_filename_2 = 0;
	const char* point_cloud_filename = 0;

	// Elementes for the filling option -- fill the empty disparity pixels with LIDAR information
	const char* improvement = "--improvement=";
	enum {NO_FILL = 0, FILL = 1};
	int imp = 0;

	// Elementes for the filling option -- fill the empty disparity pixels with LIDAR information
	const char* thread = "--thread=";
	enum {SINGLE = 0, MULTI = 1};
	int thr = 0;

	for( int i = 1; i < argc; i++ )
	{
		if( argv[i][0] != '-' )
		{
			if( !img1_filename )
				img1_filename = argv[i];
			else
				img2_filename = argv[i];
		}
		else if( strncmp(argv[i], improvement, strlen(improvement)) == 0 )
		{
			char* _imp = argv[i] + strlen(improvement);
			 imp = strcmp(_imp, "no_fill") == 0 ? NO_FILL :
					strcmp(_imp, "fill") == 0 ? FILL : -1;
			if( imp < 0 )
			{
				printf("Command-line parameter error: Unknown improvement option\n\n");
				print_help();
				return -1;
			}
		}
		else if(strncmp(argv[i], thread, strlen(thread)) == 0 )
		{
			char* _thr = argv[i] + strlen(thread);
			 thr = strcmp(_thr, "single") == 0 ? SINGLE :
					strcmp(_thr, "multi") == 0 ? MULTI : -1;
			if( imp < 0 )
			{
				printf("Command-line parameter error: Unknown thread option\n\n");
				print_help();
				return -1;
			}
		}
	}

	if( !img1_filename || !img2_filename )
	{
		printf("Command-line parameter error: both left and right images must be specified\n");
		return -1;
	}

	// Select threading option -- single or multi
	if(thr==SINGLE)
	{
		cv::setNumThreads(0);
		cout<<"Multithreading OFF"<<endl;
	}
	else
	{
		cv::setNumThreads(1);
		cout<<"Multithreading ON"<<endl;
	}

	intrinsic_filename = "stereo_parameters_new/int.yml";
	extrinsic_filename = "stereo_parameters_new/ent.yml";
	disparity_filename1 = "DISP1.png";
	disparity_filename2 = "DISP2.png";
	point_cloud_filename = "stereo_out.pcd";

		int iterator=0;

		enum { STEREO_BM=0 };
		int alg = STEREO_BM;
		int SADWindowSize = 0, numberOfDisparities = 0, min_Disparities = 0;
		bool no_display = false;
		float scale = 1.f; //0.5;

		int color_mode = alg == STEREO_BM ? 0 : -1;
		cv::Mat img1 = cv::imread(img1_filename, color_mode);
		cv::Mat img2 = cv::imread(img2_filename, color_mode);

		cv::Mat img_colored = cv::imread(img1_filename, -1);

		if( scale != 1.f )
		{
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

			fs["R"] >> R;
			fs["T"] >> T;

			stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, 0, 0, img_size, &roi1, &roi2 );

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

			cv::imwrite("left_rect.png",img1);
			cv::imwrite("right_rect.png",img2);

			//cout<<"R1:"<<endl<<R1<<endl;
		}

		LEFT = img1;
		RIGHT = img2;

	     //// 1) READ THE LIDAR POINT CLOUD AND EXTRACT THE POINTS
		pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud (new pcl::PointCloud<pcl::PointXYZ>);
		//// Loading the point clouds
		if (pcl::io::loadPCDFile<pcl::PointXYZ> ("lidar.pcd", *lidar_cloud) == -1) //* load the file
			{
				PCL_ERROR ("Couldn't read the LIDAR point cloud. \n");
				return (-1);
			}

		//// The Transform Matrix which moves the lidar points in the stereo camera frame
		//// This matrix comes from the ICP algorithm appliead before-hand
		Eigen::Matrix4f transform_lTs = Eigen::Matrix4f::Identity();
		transform_lTs(0,0) = -0.0298508; //-0.0277619;
		transform_lTs(0,1) =  0.999563; //0.99962;
		transform_lTs(0,2) = -0.00252324; //-0.000680065;
		transform_lTs(0,3) = 0.752321; //0.763683;

		transform_lTs(1,0) = 0.242289; //0.242838;
		transform_lTs(1,1) = 0.00478399; //0.00608248;
		transform_lTs(1,2) =  -0.970206; //-0.970079;
		transform_lTs(1,3) = 0.128226; //0.130637;

		transform_lTs(2,0) = -0.969755; //-0.969682;
		transform_lTs(2,1) = -0.0295702; //-0.0270913;
		transform_lTs(2,2) = -0.242321; //-0.242909;
		transform_lTs(2,3) = 2.05342; //2.06299;


		//// This is the R matrix from the calibration process - it will overlap the LIDAR onto the STEREO view
		Eigen::Matrix4f transform_lTs_R = Eigen::Matrix4f::Identity();
		transform_lTs_R(0,0) = 0.9999898012149877; //0.9998742815117727;
		transform_lTs_R(0,1) = -0.004498446772515268; //-0.005362130892552267;
		transform_lTs_R(0,2) =	-0.0004017992584439221; //0.01492208844658231;

		transform_lTs_R(1,0) = 0.004497161244118079; //0.005336969170805187;
		transform_lTs_R(1,1) = 0.9999849421569889; //0.9999842695505317;
		transform_lTs_R(1,2) = -0.003144996029905281; //0.001725517765564668;

		transform_lTs_R(2,0) = 0.0004159408054540794; //-0.01493110616754043;
		transform_lTs_R(2,1) = 0.00314315699871394; //-0.001645662110076336;
		transform_lTs_R(2,2) = 0.9999949737660324; //0.999887170567176;

		//cout<<"T2:"<<endl<<transform_lTs_R<<endl;

		//// Executing the transformation
		pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
		pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud2 (new pcl::PointCloud<pcl::PointXYZ> ());
		//// You can either apply transform_1 or transform_2; they are the same
		pcl::transformPointCloud (*lidar_cloud, *transformed_cloud, transform_lTs);
		pcl::transformPointCloud (*transformed_cloud, *transformed_cloud2, transform_lTs_R);

		pcl::PointCloud<pcl::PointXYZ> Lcloud=*transformed_cloud2;
		pcl::PointCloud<pcl::PointXYZ> saved_cloud;
		saved_cloud.height=Lcloud.height;
		saved_cloud.width=Lcloud.width;

		for(int s=0;s<saved_cloud.points.size(); ++s)
			{
				saved_cloud.points[s].x=0;
				saved_cloud.points[s].y=0;
				saved_cloud.points[s].z=0;
			}

		//// Create empty disparity map for Lidar data
		cv::Mat lidar_l=cv::Mat::zeros(img1.rows, img1.cols, CV_16S);

		float f = P1.at<double>(0,0); // Focal length
		float B = P2.at<double>(0,3)/f; // Baseline in the x direction
		float cx = P1.at<double>(0,2); // Center x coordinate
		float cy = P1.at<double>(1,2); // Center y coordinate

		float cx2 = P2.at<double>(0,2); // Center x coordinate of right image
		float cy2 = P2.at<double>(1,2); // Center y coordinate of right image
		float dcx = cx-cx2;

		for (int i = 0; i < Lcloud.points.size (); ++i)  //try with cloud height and width
		{
			//// Cross reference the 2D points
			double x_=806.6065420854103*Lcloud.points[i].x+982.1147003173828*Lcloud.points[i].z;
			double y_=806.6065420854103*Lcloud.points[i].y+515.74658203125*Lcloud.points[i].z;
			double z_=Lcloud.points[i].z;

			//// If x_2d and y_2d match x and y (from below), than the tranformation was correctly performed
			int x_2d=x_/z_;
			int y_2d=y_/z_;

			//// Scale position from m to mm;
			Lcloud.points[i].x = Lcloud.points[i].x/0.001;
			Lcloud.points[i].y = Lcloud.points[i].y/0.001;
			Lcloud.points[i].z = Lcloud.points[i].z/0.001;

			float d = (Lcloud.points[i].z*dcx-f*B)/Lcloud.points[i].z; // disparity
			float W = B/(-d+dcx); // Weighting
			int x = (Lcloud.points[i].x+cx*W)/W; // x value
			int y = (Lcloud.points[i].y+cy*W)/W; // y value

			//// Filter out all LIDAR points which are outside the camera view
			if(y>=0 && y<lidar_l.rows && x>=0 && x<lidar_l.cols)
				{
					lidar_l.at<int16_t>(y,x)=d;
					//std::cout<<"x: "<<x<<" y: "<<y<<" d: "<<d<<endl;
					saved_cloud.points.push_back(Lcloud.points[i]);
				}
		}

		saved_cloud.width = 1;
		saved_cloud.height = saved_cloud.points.size();

		pcl::io::savePCDFileASCII ("saved_lidar.pcd", saved_cloud);

		//// Save result of Lidar->image
		cv::imwrite("lidar_disp.png",lidar_l);

		//// 2) USE THE LIDAR POINTS TO INFLUENCE THE STEREO_BM METHOD
		//// ACCESS THE DISPARITY POINTS
		numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

		// Initialize global LIDAR matrix
		cv::Mat lidar_DISP=cv::Mat::ones(img1.rows, img1.cols, CV_16S);

		//// Begin FLANN search
		cv::Mat lidar_points_mat = cv::Mat::zeros(2072641, 2, CV_16S);
		std::vector<cv::Point_<int16_t> > lidar_points;
		std::vector<int> lidar_point_idx_to_global_idx;
		cv::Mat stereo_points = cv::Mat::zeros(2072641, 2, CV_16S);

		int c_l = 0;
		int c_s = 0;

		for(int w = 0; w < lidar_l.rows; ++w) {
			for(int v = 0; v < lidar_l.cols; ++v) {
				if(lidar_l.at<int16_t>(w,v)!=0)
					{
						lidar_points_mat.at<int16_t>(c_l,0)=v;
						lidar_points_mat.at<int16_t>(c_l,1)=w;
						lidar_points.push_back(cv::Point_<int16_t>(v,w)); //Stupid CV points need (x,y), NOT (row,col)
						lidar_point_idx_to_global_idx.push_back(c_l);
						c_l+=1;
					}
			}
		}

		for(int w = 0; w < img1.rows; ++w) {
				for(int v = 0; v < img1.cols; ++v) {
				if(img1.at<int16_t>(w,v)!=0) //STEREO.at<int16_t>(w,v)!=0
					{
					stereo_points.at<int16_t>(c_s,0)=v; //cols alias x
					stereo_points.at<int16_t>(c_s,1)=w; //rows alias y
					c_s+=1;
					}
				}
			}

		//std::cout<<"c_l: "<<c_l<<" c_s: "<<c_s<<endl;

		//// Convert your 2D lidar points to FLANN matrix
		flann::Matrix<int16_t> lidar_points_flann(reinterpret_cast<int16_t*>(&lidar_points[0]), lidar_points.size(), 2);
		// Create single k-d tree
		flann::KDTreeSingleIndex<flann::L2_Simple<int16_t> > kdtree_flann(lidar_points_flann);    //<-------------------------------- DISTANCE FOR FLANN SEARCH
		kdtree_flann.buildIndex();

		//// Convert the 2D stereo points to FLANN
		flann::Matrix<int16_t> stereo_points_flann(stereo_points.ptr<int16_t>(), stereo_points.rows, stereo_points.cols);
		//// Do search
		vector<vector<size_t> > indices_flann;
		vector<vector<float> > dists_flann;

	    struct timeval start_f, end_f;
	    long mtime_f, seconds_f, useconds_f;
	    gettimeofday(&start_f, NULL);

		kdtree_flann.knnSearch(stereo_points_flann, indices_flann, dists_flann, 1, flann::SearchParams());

		gettimeofday(&end_f, NULL);
	    seconds_f  = end_f.tv_sec  - start_f.tv_sec;
	    useconds_f = end_f.tv_usec - start_f.tv_usec;
	    mtime_f = ((seconds_f) * 1000 + useconds_f/1000.0) + 0.5;
	    printf("Elapsed time for FLANN: %f seconds\n", (float)mtime_f/1000);


		//// Counters for keeping track of position within the STEREO left image and the LIDAR image
		int s_i, s_j = 0;
		int l_i, l_j = 0;

		for(int ot=0;ot<indices_flann.size();++ot){
			for(int in=0;in<indices_flann.at(ot).size();in++){
				//cout<< stereo_points.row(ot) <<" ("<<indices_flann.at(ot).at(in)<<","<<dists_flann.at(ot).at(in)<<") " << lidar_points.row(indices_flann.at(ot).at(in))<<endl;
				//cout<<"Stereo: "<< stereo_points.row(ot) <<" -> Lidar:"<< lidar_points.row(indices_flann.at(ot).at(in))<<endl;
				s_i=stereo_points.row(ot).at<int16_t>(0,0); //row
				s_j=stereo_points.row(ot).at<int16_t>(0,1); //cols

				l_i=lidar_points_mat.row( lidar_point_idx_to_global_idx.at(indices_flann.at(ot).at(in)) ).at<int16_t>(0,0);
				l_j=lidar_points_mat.row( lidar_point_idx_to_global_idx.at(indices_flann.at(ot).at(in)) ).at<int16_t>(0,1);

				//cout<<"s_i: "<< s_i<<"  s_j: "<< s_j<<"  l_i: "<< l_i<<"  l_j: "<< l_j<<endl;
				if(abs(s_i-l_i)<20 && abs(s_j-l_j)<20)
					{
						lidar_DISP.at<int16_t>(s_j,s_i) = lidar_l.at<int16_t>(l_j,l_i);
					}
				////assign to each pixel in the STEREO left image the disparity of the corresponding nearest point from the LIDAR image
				//lidar_DISP.at<int16_t>(s_j,s_i) = lidar_l.at<int16_t>(l_j,l_i);
			}
		}

		DISP = lidar_DISP;

		/*
		//// Normalize the disparity map (for viewing)
		cv::Mat temp;
		cv::normalize(DISP,temp,0,255,cv::NORM_MINMAX,CV_8U);
		imshow("disp",temp);cv::waitKey(); */

		cv::StereoBM bm;

		bm.state->roi1 = roi1;
		bm.state->roi2 = roi2;
		bm.state->preFilterCap = 63;
		bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
		bm.state->minDisparity = min_Disparities;
		bm.state->numberOfDisparities = numberOfDisparities;
		bm.state->textureThreshold = 10;
		bm.state->uniquenessRatio = 15;  //5-15
		bm.state->speckleWindowSize = 100; //50-200
		bm.state->speckleRange = 32;
		bm.state->disp12MaxDiff = 1; // positive
		int cn = img1.channels();
		INTERVAL = 60; // changes the search range inside the BM method

	    struct timeval start, end;
	    long mtime, seconds, useconds;
	    gettimeofday(&start, NULL);

		bm(img1, img2, disp); // <-----------------------------------

		gettimeofday(&end, NULL);
	    seconds  = end.tv_sec  - start.tv_sec;
	    useconds = end.tv_usec - start.tv_usec;
	    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	    printf("Elapsed time for BM: %f seconds\n", (float)mtime/1000);


		disp = disp/16; //because it is multiplied by 16 inside the BM method!!

		if(disparity_filename1)
			imwrite(disparity_filename1, disp);


		//// Infill the disparity
		////  Fill in the -1 pixels with Lidar points
		if(imp==FILL){
			cout<<"Filling disparity with LIDAR information..."<<endl;
			for(int w = 0; w < disp.rows; ++w) {
						for(int v = 0; v < disp.cols; ++v) {
							if(disp.at<int16_t>(w,v)==-1 && DISP.at<int16_t>(w,v)>1)
								{
									disp.at<int16_t>(w,v) = DISP.at<int16_t>(w,v);
									//cout<<"filled in disp:"<<disp.at<int16_t>(w,v)<<endl;
								}
							}
						}

					if(disparity_filename2)
						imwrite(disparity_filename2, disp);
		}

		if(point_cloud_filename) {
			printf("Storing the point cloud...");
			cout<<endl;
			fflush(stdout);
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
			out->height = disp.rows;
			out->width = disp.cols;
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

					float d = float(disp.at<int16_t>(y,x)); // Disparity
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

					out->at(x,y) = point;
				}
			}
			pcl::io::savePCDFile(point_cloud_filename, *out);

			//saveXYZ(point_cloud_filename, xyz);
			//showXYZ(xyz, img_colored, point_cloud_filename);
			printf("\n");
		}


		printf("PROGRAM DONE!! \n");
		return 0;
}
