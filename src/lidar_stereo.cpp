#include "lidar_stereo.h"

int main (int argc, char** argv)
{

	const char* algorithm_opt = "--algorithm=";
	const char* maxdisp_opt = "--max-disparity=";
	const char* mindisp_opt = "--min-disparity=";
	const char* blocksize_opt = "--blocksize=";
	const char* nodisplay_opt = "--no-display=";
	const char* scale_opt = "--scale=";

	const char* img1_filename = 0; //left image
	const char* img2_filename = 0; //right image
	const char* intrinsic_filename = 0; //intrinsic parameters
	const char* extrinsic_filename = 0; //extrinsic parameters
	const char* disparity_filename1 = 0;
	const char* disparity_filename2 = 0;
	const char* experiment_filename_1 = 0;
	const char* experiment_filename_2 = 0;
	const char* point_cloud_filename = 0;

	cv::setNumThreads(0);

	img1_filename = "left.jpg";
	img2_filename = "right.jpg";
	intrinsic_filename = "stereo_parameters/int.yml";
	extrinsic_filename = "stereo_parameters/ent.yml";
	disparity_filename1 = "DISP1.png";
	disparity_filename2 = "DISP2.png";

		int iterator=0;

		enum { STEREO_BM=0 };
		int alg = STEREO_BM;
		int SADWindowSize = 0, numberOfDisparities = 0, min_Disparities = 0;
		bool no_display = false;
		float scale = 1.f;

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
		transform_lTs(0,0) = -0.0282604;
		transform_lTs(0,1) = 0.997477;
		transform_lTs(0,2) = 0.0656693;
		transform_lTs(0,3) = 0.815621;

		transform_lTs(1,0) = 0.136367;
		transform_lTs(1,1) = 0.0689264;
		transform_lTs(1,2) = -0.988274;
		transform_lTs(1,3) = 0.049019;

		transform_lTs(2,0) = -0.990292;
		transform_lTs(2,1) = -0.0189803;
		transform_lTs(2,2) = -0.137977;
		transform_lTs(2,3) = 2.12426;


		//// This is the R matrix from the calibration process - it will overlap the LIDAR onto the STEREO view
		Eigen::Matrix4f transform_lTs_R = Eigen::Matrix4f::Identity();
		transform_lTs_R(0,0) = 0.9998742815117727;
		transform_lTs_R(0,1) =  -0.005362130892552267;
		transform_lTs_R(0,2) =	0.01492208844658231;

		transform_lTs_R(1,0) = 0.005336969170805187;
		transform_lTs_R(1,1) = 0.9999842695505317;
		transform_lTs_R(1,2) = 0.001725517765564668;

		transform_lTs_R(2,0) = -0.01493110616754043;
		transform_lTs_R(2,1) = -0.001645662110076336;
		transform_lTs_R(2,2) = 0.999887170567176;

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
					lidar_points.push_back(cv::Point_<int16_t>(v,w));//Stupid CV points need (x,y), NOT (row,col)
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
		flann::KDTreeSingleIndex<flann::L1<int16_t> > kdtree_flann(lidar_points_flann);
		kdtree_flann.buildIndex();

		//// Convert the 2D stereo points to FLANN
		flann::Matrix<int16_t> stereo_points_flann(stereo_points.ptr<int16_t>(), stereo_points.rows, stereo_points.cols);
		//// Do search
		vector<vector<size_t> > indices_flann;
		vector<vector<float> > dists_flann;
		int64 t_F = cv::getTickCount();
		kdtree_flann.knnSearch(stereo_points_flann, indices_flann, dists_flann, 1, flann::SearchParams());
	    //cout<<"check5"<<endl;
		t_F = cv::getTickCount() - t_F;
		printf("Time elapsed for FLANN: %fms\n", t_F*1000/cv::getTickFrequency());

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

				////assign to each pixel in the STEREO left image the disparity of the corresponding nearest point from the LIDAR image
				lidar_DISP.at<int16_t>(s_j,s_i) = lidar_l.at<int16_t>(l_j,l_i);

				/*if(s_i >500 & s_j>500)
				{
					cv::circle( LEFT, cv::Point( s_j, s_i ), 10.0, cv::Scalar( 255, 128, 0  ), 1, 8 );
					imshow("Image",LEFT);

					cv::Mat temp;
					cv::normalize(lidar_l,temp,0,255,cv::NORM_MINMAX,CV_8U);

					cv::circle( temp, cv::Point( l_j, l_i ), 10.0, cv::Scalar( 255, 128, 0 ), 1, 8 );
					imshow("Disp",temp);
					cv::waitKey(0);
				} */
			}
		}

		DISP = lidar_DISP;

		/*
		//// Normalize the disparity map (for viewing)
		cv::Mat temp;
		cv::normalize(lidar_l,temp,0,255,cv::NORM_MINMAX,CV_8U);
		*/

		cv::StereoBM bm;

		bm.state->roi1 = roi1;
		bm.state->roi2 = roi2;
		bm.state->preFilterCap = 31;
		bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
		bm.state->minDisparity = min_Disparities;
		bm.state->numberOfDisparities = numberOfDisparities;
		bm.state->textureThreshold = 10;
		bm.state->uniquenessRatio = 15;  //5-15
		bm.state->speckleWindowSize = 2; //50-200
		bm.state->speckleRange = 1;
		bm.state->disp12MaxDiff = 10; // positive
		int cn = img1.channels();
		INTERVAL = 30;
		int64 t_bm = cv::getTickCount();
		bm(img1, img2, disp);
		t_bm = cv::getTickCount() - t_F;
		printf("Time elapsed for BM: %fms\n", t_bm*1000/cv::getTickFrequency());

		disp.convertTo(disp8, CV_16S, 255/(numberOfDisparities*16.));
		if(disparity_filename1)
			imwrite(disparity_filename1, disp8);

		//////////////////////////////////////////////////////////////////////////////////////////////

		/*cv::StereoBM second;

		second.state->roi1 = roi1;
		second.state->roi2 = roi2;
		second.state->preFilterCap = 31;
		second.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
		second.state->minDisparity = min_Disparities;
		second.state->numberOfDisparities = numberOfDisparities;
		second.state->textureThreshold = 10;
		second.state->uniquenessRatio = 15;  //5-15
		second.state->speckleWindowSize = 2; //50-200
		second.state->speckleRange = 1;
		second.state->disp12MaxDiff = 10; // positive
		cn = img1.channels();
		INTERVAL = 50;
		int64 t_second = cv::getTickCount();
		second(img1, img2, disp2);
		t_second = cv::getTickCount() - t_F;
		printf("Time elapsed for BM_second: %fms\n", t_second*1000/cv::getTickFrequency());

		disp2.convertTo(disp2_8, CV_16S, 255/(numberOfDisparities*16.));
		if(disparity_filename2)
			imwrite(disparity_filename2, disp2_8);

		cv::Mat Diff;
		Diff=cv::Mat::zeros(disp8.rows, disp8.cols, CV_16S);

		int SSD = 0;
		int diff = 0;
		for(int w = 0; w < disp8.rows; ++w) {
			for(int v = 0; v < disp8.cols; ++v) {
				if(disp8.at<int16_t>(w,v)!=0 && disp2_8.at<int16_t>(w,v)!=0 && disp8.at<int16_t>(w,v)!=disp2_8.at<int16_t>(w,v)) //
					{
						cout<<"disp1: "<<disp8.at<int16_t>(w,v)<<endl;
						cout<<"disp2: "<<disp2_8.at<int16_t>(w,v)<<endl;

						diff = abs(disp8.at<int16_t>(w,v)) - abs(disp2_8.at<int16_t>(w,v));
						SSD += diff*diff;

						cout<<"diff: "<<diff<<endl;
						cout<<endl;

						Diff.at<int16_t>(w,v)=100;
					}
				}
			}

		imwrite("Diff.png", Diff);

		cout<<"SSD: "<<SSD<<endl; */

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		for(int w = 0; w < disp8.rows; ++w) {
			for(int v = 0; v < disp8.cols; ++v) {
				if(disp8.at<int16_t>(w,v)==-1)
					{
						disp8.at<int16_t>(w,v) = 0;
					}
				}
			}


		for(int w = 0; w < disp8.rows; ++w) {
			for(int v = 0; v < disp8.cols; ++v) {
				//cout<<"disp8: "<<disp8.at<int16_t>(w,v)<<endl;
				if(disp8.at<int16_t>(w,v)==0)
					{
					    int st_disp = abs(disp8.at<int16_t>(w,v));
					    int ld_disp = abs(DISP.at<int16_t>(w,v));
					    /*cout<<"disp8: "<< st_disp <<endl;
					    cout<<"DISP: "<< ld_disp <<endl;
					    cout<<"diff: "<< st_disp - ld_disp <<endl;
					    cout<<endl; */
					    if((abs(st_disp - ld_disp)<20 && abs(st_disp - ld_disp)>5) || (abs(st_disp - ld_disp)<80 && abs(st_disp - ld_disp)>65))
					    {
					    	disp8.at<int16_t>(w,v) = DISP.at<int16_t>(w,v);
					    }
					}
				}
			}


		imwrite(disparity_filename2, disp8);




		printf("PROGRAM DONE!! \n");
		return 0;
}
