
#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include "cvconfig.h"

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <vector>

#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/calib3d/calib3d_tegra.hpp"
#else
#define GET_OPTIMIZED(func) (func)
#endif

#endif
