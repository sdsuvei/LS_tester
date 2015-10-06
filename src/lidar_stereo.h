/*
 * lidar_stereo.h
 *
 *  Created on: Sep 10, 2015
 *      Author: sdsuvei
 */

#ifndef LIDAR_STEREO_H_
#define LIDAR_STEREO_H_

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "flann/flann.hpp"
#include "precomp.hpp"

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
int flag = 0;
cv::Mat disp, disp8, DISP, LEFT, RIGHT;
//int min_D = 0;
//int max_D = 0;


//convenient typedefs
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

CV_IMPL CvStereoBMState* cvCreateStereoBMState( int /*preset*/, int numberOfDisparities )
{
    CvStereoBMState* state = (CvStereoBMState*)cvAlloc( sizeof(*state) );
    if( !state )
        return 0;

    state->preFilterType = CV_STEREO_BM_XSOBEL; //CV_STEREO_BM_NORMALIZED_RESPONSE;
    state->preFilterSize = 9;
    state->preFilterCap = 31;
    state->SADWindowSize = 15;
    state->minDisparity = 0;
    state->numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : 64;
    state->textureThreshold = 10;
    state->uniquenessRatio = 15;
    state->speckleRange = state->speckleWindowSize = 0;
    state->trySmallerWindows = 0;
    state->roi1 = state->roi2 = cvRect(0,0,0,0);
    state->disp12MaxDiff = -1;

    state->preFilteredImg0 = state->preFilteredImg1 = state->slidingSumBuf =
    state->disp = state->cost = 0;

    return state;
}

CV_IMPL void cvReleaseStereoBMState( CvStereoBMState** state )
{
    if( !state )
        CV_Error( CV_StsNullPtr, "" );

    if( !*state )
        return;

    cvReleaseMat( &(*state)->preFilteredImg0 );
    cvReleaseMat( &(*state)->preFilteredImg1 );
    cvReleaseMat( &(*state)->slidingSumBuf );
    cvReleaseMat( &(*state)->disp );
    cvReleaseMat( &(*state)->cost );
    cvFree( state );
}

namespace cv
{

static void prefilterNorm( const Mat& src, Mat& dst, int winsize, int ftzero, uchar* buf )
{
    int x, y, wsz2 = winsize/2;
    int* vsum = (int*)alignPtr(buf + (wsz2 + 1)*sizeof(vsum[0]), 32);
    int scale_g = winsize*winsize/8, scale_s = (1024 + scale_g)/(scale_g*2);
    const int OFS = 256*5, TABSZ = OFS*2 + 256;
    uchar tab[TABSZ];
    const uchar* sptr = src.data;
    int srcstep = (int)src.step;
    Size size = src.size();

    scale_g *= scale_s;

    for( x = 0; x < TABSZ; x++ )
        tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero*2 : x - OFS + ftzero);

    for( x = 0; x < size.width; x++ )
        vsum[x] = (ushort)(sptr[x]*(wsz2 + 2));

    for( y = 1; y < wsz2; y++ )
    {
        for( x = 0; x < size.width; x++ )
            vsum[x] = (ushort)(vsum[x] + sptr[srcstep*y + x]);
    }

    for( y = 0; y < size.height; y++ )
    {
        const uchar* top = sptr + srcstep*MAX(y-wsz2-1,0);
        const uchar* bottom = sptr + srcstep*MIN(y+wsz2,size.height-1);
        const uchar* prev = sptr + srcstep*MAX(y-1,0);
        const uchar* curr = sptr + srcstep*y;
        const uchar* next = sptr + srcstep*MIN(y+1,size.height-1);
        uchar* dptr = dst.ptr<uchar>(y);
        x = 0;

        for( ; x < size.width; x++ )
            vsum[x] = (ushort)(vsum[x] + bottom[x] - top[x]);

        for( x = 0; x <= wsz2; x++ )
        {
            vsum[-x-1] = vsum[0];
            vsum[size.width+x] = vsum[size.width-1];
        }

        int sum = vsum[0]*(wsz2 + 1);
        for( x = 1; x <= wsz2; x++ )
            sum += vsum[x];

        int val = ((curr[0]*5 + curr[1] + prev[0] + next[0])*scale_g - sum*scale_s) >> 10;
        dptr[0] = tab[val + OFS];

        for( x = 1; x < size.width-1; x++ )
        {
            sum += vsum[x+wsz2] - vsum[x-wsz2-1];
            val = ((curr[x]*4 + curr[x-1] + curr[x+1] + prev[x] + next[x])*scale_g - sum*scale_s) >> 10;
            dptr[x] = tab[val + OFS];
        }

        sum += vsum[x+wsz2] - vsum[x-wsz2-1];
        val = ((curr[x]*5 + curr[x-1] + prev[x] + next[x])*scale_g - sum*scale_s) >> 10;
        dptr[x] = tab[val + OFS];
    }
}


static void
prefilterXSobel( const Mat& src, Mat& dst, int ftzero )
{
    int x, y;
    const int OFS = 256*4, TABSZ = OFS*2 + 256;
    uchar tab[TABSZ];
    Size size = src.size();

    for( x = 0; x < TABSZ; x++ )
        tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero*2 : x - OFS + ftzero);
    uchar val0 = tab[0 + OFS];

    for( y = 0; y < size.height-1; y += 2 )
    {
        const uchar* srow1 = src.ptr<uchar>(y);
        const uchar* srow0 = y > 0 ? srow1 - src.step : size.height > 1 ? srow1 + src.step : srow1;
        const uchar* srow2 = y < size.height-1 ? srow1 + src.step : size.height > 1 ? srow1 - src.step : srow1;
        const uchar* srow3 = y < size.height-2 ? srow1 + src.step*2 : srow1;
        uchar* dptr0 = dst.ptr<uchar>(y);
        uchar* dptr1 = dptr0 + dst.step;

        dptr0[0] = dptr0[size.width-1] = dptr1[0] = dptr1[size.width-1] = val0;
        x = 1;

        for( ; x < size.width-1; x++ )
        {
            int d0 = srow0[x+1] - srow0[x-1], d1 = srow1[x+1] - srow1[x-1],
                d2 = srow2[x+1] - srow2[x-1], d3 = srow3[x+1] - srow3[x-1];
            int v0 = tab[d0 + d1*2 + d2 + OFS];
            int v1 = tab[d1 + d2*2 + d3 + OFS];
            dptr0[x] = (uchar)v0;
            dptr1[x] = (uchar)v1;
        }
    }

    for( ; y < size.height; y++ )
    {
        uchar* dptr = dst.ptr<uchar>(y);
        x = 0;

        for(; x < size.width; x++ )
            dptr[x] = val0;
    }
}

static const int DISPARITY_SHIFT = 4;
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------------THE MAGIC HAPPENS HERE---------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
static void
findStereoCorrespondenceBM( const Mat& left, const Mat& right,
                            Mat& disp, Mat& cost, const CvStereoBMState& state,
                            uchar* buf, int _dy0, int _dy1 )
{
    const int ALIGN = 16;
    int x, y, d;
    int wsz = state.SADWindowSize, wsz2 = wsz/2;             // wsz = 9;   wsz2 = 4;
    int dy0 = MIN(_dy0, wsz2+1), dy1 = MIN(_dy1, wsz2+1);    // dy0 = 5; dy1 = 5;
    int ndisp = state.numberOfDisparities;                   // ndisp = 240;
    int mindisp = state.minDisparity;                        // mindisp = 0;
    int lofs = MAX(ndisp - 1 + mindisp, 0);                  // lofs = 239;
    int rofs = -MIN(ndisp - 1 + mindisp, 0);                 // rofs = 0;
    int width = left.cols, height = left.rows;				 // width = 1920; height = 78;
    //cout<<"height: "<<height<<endl;
    int width1 = width - rofs - ndisp + 1;					 // width1 = 1681;
    int ftzero = state.preFilterCap;                         // ftzero = 31;
    int textureThreshold = state.textureThreshold;           // textureThreshold = 10;
    int uniquenessRatio = state.uniquenessRatio;             // uniquenessRation = 15;
    short FILTERED = (short)((mindisp - 1) << DISPARITY_SHIFT); // FILTERED = -16;

    int *sad, *hsad0, *hsad, *hsad_sub, *htext;
    uchar *cbuf0, *cbuf;
    const uchar* lptr0 = left.data + lofs;
    const uchar* rptr0 = right.data + rofs;
    const uchar *lptr, *lptr_sub, *rptr;
    //pointer to the disparity data
    short* dptr = (short*)disp.data;
    // the step used to compute each value of Mat
    int sstep = (int)left.step;                              // sstep = 1920;
    int dstep = (int)(disp.step/sizeof(dptr[0]));            // dstep = 1920
    int cstep = (height+dy0+dy1)*ndisp;                      // cstep = 19680;
    int costbuf = 0;
    int coststep = cost.data ? (int)(cost.step/sizeof(costbuf)) : 0; // coststep = 19680;
    const int TABSZ = 256;
    uchar tab[TABSZ];

    sad = (int*)alignPtr(buf + sizeof(sad[0]), ALIGN); // (12,ALIGN)
    hsad0 = (int*)alignPtr(sad + ndisp + 1 + dy0*ndisp, ALIGN); //(1445,ALIGN)
    htext = (int*)alignPtr((int*)(hsad0 + (height+dy1)*ndisp) + wsz2 + 2, ALIGN); //(19930,ALIGN)
    cbuf0 = (uchar*)alignPtr((uchar*)(htext + height + wsz2 + 2) + dy0*ndisp, ALIGN); //(88,ALIGN);


    for( x = 0; x < TABSZ; x++ ) //TABSZ=256
        tab[x] = (uchar)std::abs(x - ftzero); //ftzero=31 tab goes from 31 to 225

    memset( hsad0 - dy0*ndisp, 0, (height + dy0 + dy1)*ndisp*sizeof(hsad0[0]) ); //   ( ADDRESS, 0, 83520)
    memset( htext - wsz2 - 1, 0, (height + wsz + 1)*sizeof(htext[0]) );          //   ( ADDRESS, 0, 348)

    for( x = -wsz2-1; x < wsz2; x++ ) // from -5 to 4 <---- the S.A.D window size
    {
        hsad = hsad0 - dy0*ndisp; 												// hsad0+1200
        cbuf = cbuf0 + (x + wsz2 + 1)*cstep - dy0*ndisp;						// cbuf = cbuf0+(x+5)*19680-1200
        lptr = lptr0 + std::min(std::max(x, -lofs), width-lofs-1) - dy0*sstep;  // lptr0+x-9600=left.data+x+239-9600
        rptr = rptr0 + std::min(std::max(x, -rofs), width-rofs-1) - dy0*sstep;  // rptr0+x-9600=right.data+x-9600

        // from -4 to 82, hsad+240; cbuf+240; lptr+1920; rptr+1920
        for( y = -dy0; y < height + dy1; y++, hsad += ndisp, cbuf += ndisp, lptr += sstep, rptr += sstep )
        {
            int lval = lptr[0]; //

            for( d = 0; d < ndisp; d++ )
            {
                int diff = std::abs(lval - rptr[d]); //difference between left pixel and right pixel
                cbuf[d] = (uchar)diff;  //this is the cost buffer
                hsad[d] = (int)(hsad[d] + diff);
            }
            htext[y] += tab[lval];

        }
    }

    for( y = 0; y < height; y++ )
    {
        for( x = 0; x < lofs; x++ )
            {
        	dptr[y*dstep + x] = FILTERED;

            }

        for( x = lofs + width1; x < width; x++ )
            dptr[y*dstep + x] = FILTERED;
    }
    dptr += lofs;

    for( x = 0; x < width1; x++, dptr++ )
    {
        int* costptr = cost.data ? (int*)cost.data + lofs + x : &costbuf;
        int x0 = x - wsz2 - 1, x1 = x + wsz2;

        const uchar* cbuf_sub = cbuf0 + ((x0 + wsz2 + 1) % (wsz + 1))*cstep - dy0*ndisp;
        cbuf = cbuf0 + ((x1 + wsz2 + 1) % (wsz + 1))*cstep - dy0*ndisp;
        hsad = hsad0 - dy0*ndisp;
        lptr_sub = lptr0 + MIN(MAX(x0, -lofs), width-1-lofs) - dy0*sstep;
        lptr = lptr0 + MIN(MAX(x1, -lofs), width-1-lofs) - dy0*sstep;
        rptr = rptr0 + MIN(MAX(x1, -rofs), width-1-rofs) - dy0*sstep;

        for( y = -dy0; y < height + dy1; y++, cbuf += ndisp, cbuf_sub += ndisp, hsad += ndisp, lptr += sstep, lptr_sub += sstep, rptr += sstep )
        {
            int lval = lptr[0];
            for( d = 0; d < ndisp; d++ )
            {
                int diff = std::abs(lval - rptr[d]);
                cbuf[d] = (uchar)diff;
                hsad[d] = hsad[d] + diff - cbuf_sub[d];
            }
            htext[y] += tab[lval] - tab[lptr_sub[0]];
        }

        for( y = dy1; y <= wsz2; y++ )
        {
            htext[height+y] = htext[height+dy1-1];
        }

        for( y = -wsz2-1; y < -dy0; y++ )
        {
        	htext[y] = htext[-dy0];
        }

        for( d = 0; d < ndisp; d++ )
            sad[d] = (int)(hsad0[d-ndisp*dy0]*(wsz2 + 2 - dy0)); //(hsad0[d-1200])

        hsad = hsad0 + (1 - dy0)*ndisp; //  hsad0-960
        for( y = 1 - dy0; y < wsz2; y++, hsad += ndisp )
        {
            for( d = 0; d < ndisp; d++ )
                sad[d] = (int)(sad[d] + hsad[d]);
        }
        int tsum = 0;
        for( y = -wsz2-1; y < wsz2; y++ )
            tsum += htext[y];

        for( y = 0; y < height; y++ )
        {
            int minsad = INT_MAX, mind = -1;
            hsad = hsad0 + MIN(y + wsz2, height+dy1-1)*ndisp;
            hsad_sub = hsad0 + MAX(y - wsz2 - 1, -dy0)*ndisp;

            int lidar_D = DISP.at<int16_t>(y, x);

        	//// Draw a circle
            /*
        	cv::circle( LEFT, cv::Point( x,y ), 10.0, cv::Scalar( 0, 0, 255 ), 1, 8 );
        	imshow("Left",LEFT);

        	cv::circle( RIGHT, cv::Point( x,y ), 10.0, cv::Scalar(255, 56, 0 ), 1, 8 );
        	imshow("Right",RIGHT);

    		cv::Mat temp;
    		cv::normalize(DISP,temp,0,255,cv::NORM_MINMAX,CV_8U);

        	cv::circle( temp, cv::Point( x,y), 10.0, cv::Scalar( 255, 128, 0 ), 1, 8 );
        	imshow("Disp",temp);
        	cv::waitKey(0);*/
        	//cout<<"y:"<<y<<"  x:"<<x<<endl;

            int min_D = 0;
            int max_D = ndisp;
            if(lidar_D>0)
            {
            	min_D = ndisp-lidar_D-30;
            	max_D = ndisp-lidar_D+30;
            }

            for( d = 0; d < ndisp; d++ )
            {
                int currsad = sad[d] + hsad[d] - hsad_sub[d];

                sad[d] = currsad;
                if(d>min_D && d<max_D && currsad < minsad ) //d>min_D && d<max_D &&
                {
                    minsad = currsad;
                    mind = d;
                }
            }

            //minsad=0;
            //mind=lidar_D;

            tsum += htext[y + wsz2] - htext[y - wsz2 - 1];
            if( tsum < textureThreshold )
            {
                dptr[y*dstep] = FILTERED;
                continue;
            }

            if( uniquenessRatio > 0 )
            {
                int thresh = minsad + (minsad * uniquenessRatio/100);
                for( d = 0; d < ndisp; d++ )
                {
                    if( sad[d] <= thresh && (d < mind-1 || d > mind+1))
                        break;
                }
                if( d < ndisp )
                {
                    dptr[y*dstep] = FILTERED;
                    continue;
                }
            }

            {

            sad[-1] = sad[1];
            sad[ndisp] = sad[ndisp-2];
            int p = sad[mind+1], n = sad[mind-1];
            d = p + n - 2*sad[mind] + std::abs(p - n); //934-816+114=232
            dptr[y*dstep] = (short)(((ndisp - mind - 1 + mindisp)*256 + (d != 0 ? (p-n)*256/d : 0) + 15) >> 4);
            int value=(short)((ndisp - mind - 1 + mindisp)*256 + (d != 0 ? (p-n)*256/d : 0) + 15);
            costptr[y*coststep] = sad[mind];
            flag++;
            /*if(flag%2000==0)
            {
            	disp.convertTo(disp8, CV_32F);
				cv::normalize(disp8, disp8, 1, 0, CV_MINMAX);
				imshow("disp",disp8);waitKey();
            }*/
            }
        }
    }
    Mat temp;
	disp.convertTo(temp, CV_32F);
	cv::normalize(temp, temp, 1, 0, CV_MINMAX);
	imshow("After BM",temp);waitKey();

}
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------THE MAGIC ENDS HERE---------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------------------------------------------//


struct PrefilterInvoker
{
    PrefilterInvoker(const Mat& left0, const Mat& right0, Mat& left, Mat& right,
                     uchar* buf0, uchar* buf1, CvStereoBMState* _state )
    {
        imgs0[0] = &left0; imgs0[1] = &right0;
        imgs[0] = &left; imgs[1] = &right;
        buf[0] = buf0; buf[1] = buf1;
        state = _state;
    }

    void operator()( int ind ) const
    {
        if( state->preFilterType == CV_STEREO_BM_NORMALIZED_RESPONSE )
        {
        	//printf("Before call for prefilterNorm \n");
            prefilterNorm( *imgs0[ind], *imgs[ind], state->preFilterSize, state->preFilterCap, buf[ind] );
            //printf("After call for prefilterNorm \n\n");
        }
        else
        {
        	//printf("Before call for prefilterXSobel \n");
            prefilterXSobel( *imgs0[ind], *imgs[ind], state->preFilterCap );
            //printf("After call for prefilterXSobel \n\n");
        }
    }

    const Mat* imgs0[2];
    Mat* imgs[2];
    uchar* buf[2];
    CvStereoBMState *state;
};


struct FindStereoCorrespInvoker : ParallelLoopBody
{
    FindStereoCorrespInvoker( const Mat& _left, const Mat& _right,
                              Mat& _disp, CvStereoBMState* _state,
                              int _nstripes, int _stripeBufSize,
                              bool _useShorts, Rect _validDisparityRect )
    {
        left = &_left; right = &_right;
        disp = &_disp; state = _state;
        nstripes = _nstripes; stripeBufSize = _stripeBufSize;
        useShorts = _useShorts;
        validDisparityRect = _validDisparityRect;
    }

    void operator()( const Range& range ) const
    {
        int cols = left->cols, rows = left->rows;
        int _row0 = min(cvRound(range.start * rows / nstripes), rows);
        int _row1 = min(cvRound(range.end * rows / nstripes), rows);
        uchar *ptr = state->slidingSumBuf->data.ptr + range.start * stripeBufSize;
        int FILTERED = (state->minDisparity - 1)*16;

        Rect roi = validDisparityRect & Rect(0, _row0, cols, _row1 - _row0);
        if( roi.height == 0 )
            return;
        int row0 = roi.y;
        int row1 = roi.y + roi.height;

        Mat part;
        if( row0 > _row0 )
        {
            part = disp->rowRange(_row0, row0);
            part = Scalar::all(FILTERED);
        }
        if( _row1 > row1 )
        {
            part = disp->rowRange(row1, _row1);
            part = Scalar::all(FILTERED);
        }

        Mat left_i = left->rowRange(row0, row1);
        Mat right_i = right->rowRange(row0, row1);
        Mat disp_i = disp->rowRange(row0, row1);
        Mat cost_i = state->disp12MaxDiff >= 0 ? Mat(state->cost).rowRange(row0, row1) : Mat();
        Mat temp;


        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //printf("Before FIRST \n");
        findStereoCorrespondenceBM( left_i, right_i, disp_i, cost_i, *state, ptr, row0, rows - row1 );
        // printf("After FIRST \n");
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    	/*disp_i.convertTo(temp, CV_32F);
		cv::normalize(temp, temp, 1, 0, CV_MINMAX);
		imshow("After CorrespondenceBM",temp);waitKey(); */

        if( state->disp12MaxDiff >= 0 )
            validateDisparity( disp_i, cost_i, state->minDisparity, state->numberOfDisparities, state->disp12MaxDiff );

        if( roi.x > 0 )
        {
            part = disp_i.colRange(0, roi.x);
            part = Scalar::all(FILTERED);
        }
        if( roi.x + roi.width < cols )
        {
            part = disp_i.colRange(roi.x + roi.width, cols);
            part = Scalar::all(FILTERED);
        }
    	/*disp_i.convertTo(temp, CV_32F);
		cv::normalize(temp, temp, 1, 0, CV_MINMAX);
		imshow("After validateDisparity",temp);waitKey(); */
    }

protected:
    const Mat *left, *right;
    Mat* disp;
    CvStereoBMState *state;

    int nstripes;
    int stripeBufSize;
    bool useShorts;
    Rect validDisparityRect;
};

static void findStereoCorrespondenceBM( const Mat& left0, const Mat& right0, Mat& disp0, CvStereoBMState* state)
{

	cout<<"findStereoBM"<<endl;

    if (left0.size() != right0.size() || disp0.size() != left0.size())
        CV_Error( CV_StsUnmatchedSizes, "All the images must have the same size" );

    if (left0.type() != CV_8UC1 || right0.type() != CV_8UC1)
        CV_Error( CV_StsUnsupportedFormat, "Both input images must have CV_8UC1" );

    if (disp0.type() != CV_16SC1 && disp0.type() != CV_32FC1)
        CV_Error( CV_StsUnsupportedFormat, "Disparity image must have CV_16SC1 or CV_32FC1 format" );

    if( !state )
        CV_Error( CV_StsNullPtr, "Stereo BM state is NULL." );

    if( state->preFilterType != CV_STEREO_BM_NORMALIZED_RESPONSE && state->preFilterType != CV_STEREO_BM_XSOBEL )
        CV_Error( CV_StsOutOfRange, "preFilterType must be = CV_STEREO_BM_NORMALIZED_RESPONSE" );

    if( state->preFilterSize < 5 || state->preFilterSize > 255 || state->preFilterSize % 2 == 0 )
        CV_Error( CV_StsOutOfRange, "preFilterSize must be odd and be within 5..255" );

    if( state->preFilterCap < 1 || state->preFilterCap > 63 )
        CV_Error( CV_StsOutOfRange, "preFilterCap must be within 1..63" );

    if( state->SADWindowSize < 5 || state->SADWindowSize > 255 || state->SADWindowSize % 2 == 0 ||
        state->SADWindowSize >= min(left0.cols, left0.rows) )
        CV_Error( CV_StsOutOfRange, "SADWindowSize must be odd, be within 5..255 and be not larger than image width or height" );

    if( state->numberOfDisparities <= 0 || state->numberOfDisparities % 16 != 0 )
        CV_Error( CV_StsOutOfRange, "numberOfDisparities must be positive and divisble by 16" );

    if( state->textureThreshold < 0 )
        CV_Error( CV_StsOutOfRange, "texture threshold must be non-negative" );

    if( state->uniquenessRatio < 0 )
        CV_Error( CV_StsOutOfRange, "uniqueness ratio must be non-negative" );

    if( !state->preFilteredImg0 || state->preFilteredImg0->cols * state->preFilteredImg0->rows < left0.cols * left0.rows )
    {
        cvReleaseMat( &state->preFilteredImg0 );
        cvReleaseMat( &state->preFilteredImg1 );
        cvReleaseMat( &state->cost );

        state->preFilteredImg0 = cvCreateMat( left0.rows, left0.cols, CV_8U );
        state->preFilteredImg1 = cvCreateMat( left0.rows, left0.cols, CV_8U );
        state->cost = cvCreateMat( left0.rows, left0.cols, CV_16S );
    }
    Mat left(left0.size(), CV_8U, state->preFilteredImg0->data.ptr);
    Mat right(right0.size(), CV_8U, state->preFilteredImg1->data.ptr);

    int mindisp = state->minDisparity;
    int ndisp = state->numberOfDisparities;

    int width = left0.cols;
    int height = left0.rows;
    int lofs = max(ndisp - 1 + mindisp, 0);
    int rofs = -min(ndisp - 1 + mindisp, 0);
    int width1 = width - rofs - ndisp + 1;
    int FILTERED = (state->minDisparity - 1) << DISPARITY_SHIFT;

    if( lofs >= width || rofs >= width || width1 < 1 )
    {
        disp0 = Scalar::all( FILTERED * ( disp0.type() < CV_32F ? 1 : 1./(1 << DISPARITY_SHIFT) ) );
        return;
    }

    //cout<<"Disp0: \n"<<disp0<<endl;

    Mat disp = disp0;
    if( disp0.type() == CV_32F)
    {
        if( !state->disp || state->disp->rows != disp0.rows || state->disp->cols != disp0.cols )
        {
            cvReleaseMat( &state->disp );
            state->disp = cvCreateMat(disp0.rows, disp0.cols, CV_16S);
        }
        disp = cv::cvarrToMat(state->disp);
    }

    //cout<<"Disp: \n"<<disp<<endl;

    int wsz = state->SADWindowSize;
    int bufSize0 = (int)((ndisp + 2)*sizeof(int));
    bufSize0 += (int)((height+wsz+2)*ndisp*sizeof(int));
    bufSize0 += (int)((height + wsz + 2)*sizeof(int));
    bufSize0 += (int)((height+wsz+2)*ndisp*(wsz+2)*sizeof(uchar) + 256);

    int bufSize1 = (int)((width + state->preFilterSize + 2) * sizeof(int) + 256);
    int bufSize2 = 0;
    if( state->speckleRange >= 0 && state->speckleWindowSize > 0 )
        bufSize2 = width*height*(sizeof(cv::Point_<short>) + sizeof(int) + sizeof(uchar));

    bool useShorts = state->preFilterCap <= 31 && state->SADWindowSize <= 21 && checkHardwareSupport(CV_CPU_SSE2);


    const double SAD_overhead_coeff = 10.0;
    double N0 = 8000000 / (useShorts ? 1 : 4);  // approx tbb's min number instructions reasonable for one thread
    double maxStripeSize = min(max(N0 / (width * ndisp), (wsz-1) * SAD_overhead_coeff), (double)height);
    int nstripes = cvCeil(height / maxStripeSize);

    int bufSize = max(bufSize0 * nstripes, max(bufSize1 * 2, bufSize2));

    if( !state->slidingSumBuf || state->slidingSumBuf->cols < bufSize )
    {
        cvReleaseMat( &state->slidingSumBuf );
        state->slidingSumBuf = cvCreateMat( 1, bufSize, CV_8U );
    }

    uchar *_buf = state->slidingSumBuf->data.ptr;
    int idx[] = {0,1};
     printf("Before call for PrefilterInvoker \n ");
    parallel_do(idx, idx+2, PrefilterInvoker(left0, right0, left, right, _buf, _buf + bufSize1, state));
    // printf("After call for PrefilterInvoker \n\n ");
    Rect validDisparityRect(0, 0, width, height), R1 = state->roi1, R2 = state->roi2;
    validDisparityRect = getValidDisparityROI(R1.area() > 0 ? Rect(0, 0, width, height) : validDisparityRect,
                                              R2.area() > 0 ? Rect(0, 0, width, height) : validDisparityRect,
                                              state->minDisparity, state->numberOfDisparities,
                                              state->SADWindowSize);

    //printf("Before call for FindStereoCorrespInvoker \n ");
    parallel_for_(Range(0, nstripes),
                  FindStereoCorrespInvoker(left, right, disp, state, nstripes,
                                           bufSize0, useShorts, validDisparityRect));
    //printf("After call for FindStereoCorrespInvoker \n ");

    //-------------------------------------------------------------------------------------------------------------------------------------------------------

    Mat temp;
    /*
	disp.convertTo(temp, CV_32F);
	cv::normalize(temp, temp, 1, 0, CV_MINMAX);
	imshow("After Invoker",temp);waitKey();*/
    if( state->speckleRange >= 0 && state->speckleWindowSize > 0 )
    {
        Mat buf(state->slidingSumBuf);
        filterSpeckles(disp, FILTERED, state->speckleWindowSize, state->speckleRange, buf);
    }

    if (disp0.data != disp.data)
        disp.convertTo(disp0, disp0.type(), 1./(1 << DISPARITY_SHIFT), 0);

	disp.convertTo(temp, CV_32F);
	cv::normalize(temp, temp, 1, 0, CV_MINMAX);
	imshow("After Spekle",temp);waitKey();
}

StereoBM::StereoBM()
{	state = cvCreateStereoBMState(); }

StereoBM::StereoBM(int _preset, int _ndisparities, int _SADWindowSize)
{ 	init(_preset, _ndisparities, _SADWindowSize); }

void StereoBM::init(int _preset, int _ndisparities, int _SADWindowSize)
{
    state = cvCreateStereoBMState(_preset, _ndisparities);
    state->SADWindowSize = _SADWindowSize;
}

void StereoBM::operator()( InputArray _left, InputArray _right,
                           OutputArray _disparity, int disptype )
{
	//printf("In operator. \n");
    Mat left = _left.getMat(), right = _right.getMat();
    CV_Assert( disptype == CV_16S || disptype == CV_32F );
    _disparity.create(left.size(), disptype);
    Mat disparity = _disparity.getMat();
	//printf("Before call for findStereoCorrespondenceBM - SECOND \n");
    findStereoCorrespondenceBM(left, right, disparity, state);
    //printf("After call for findStereoCorrespondenceBM - SECOND \n\n");
}

template<> void Ptr<CvStereoBMState>::delete_obj()
{ cvReleaseStereoBMState(&obj); }

}

CV_IMPL void cvFindStereoCorrespondenceBM( const CvArr* leftarr, const CvArr* rightarr,
                                           CvArr* disparr, CvStereoBMState* state )
{
    cv::Mat left = cv::cvarrToMat(leftarr),
        right = cv::cvarrToMat(rightarr),
        disp = cv::cvarrToMat(disparr);
    cv::findStereoCorrespondenceBM(left, right, disp, state);
}


#endif /* LIDAR_STEREO_H_ */
