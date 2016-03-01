////////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//
///

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include <future>

using namespace std;
using namespace cv;
using namespace cv::detail;

// Default command line args
bool preview = false;
bool try_gpu = false;
double work_megapix = 0.12;
double seam_megapix = 0.12;
double compose_megapix = -1;
float conf_thresh = 0.5f;
string features_type = "surf";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "plane";
int expos_comp_type = ExposureCompensator::GAIN;
float match_conf = 0.3f;
string seam_find_type = "dp_colorgrad";
int blend_type = Blender::FEATHER;
float blend_strength = 5;
string result_name = "temp.jpg";


bool stitch(vector<Mat> orig_images) {
// Check if have enough images
	try {
		int num_images = static_cast<int>(orig_images.size());
		if (num_images < 1)
		{
			cout << "Not enough images to stitch." << endl;
			return false;
		}

		if (num_images < 2) {
			imwrite(result_name, orig_images[0]);
			return true;
		}

		double work_scale = 1, seam_scale = 1, compose_scale = 1;
		bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

		LOGLN("Finding features...");
#if ENABLE_LOG
		int64 t = getTickCount();
#endif

		Ptr<FeaturesFinder> finder;
		if (features_type == "surf")
		{
#if defined(HAVE_OPENCV_NONFREE) && defined(HAVE_OPENCV_GPU)
			if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
				finder = new SurfFeaturesFinderGpu();
			else
#endif
				finder = new SurfFeaturesFinder();
		}
		else if (features_type == "orb")
		{
			finder = new OrbFeaturesFinder();
		}
		else
		{
			cout << "Unknown 2D features type: '" << features_type << "'.\n";
			return false;
		}

		vector<Mat> full_img(num_images);
		vector<Mat> img(num_images);
		vector<ImageFeatures> features(num_images);
		vector<Mat> images(num_images);
		vector<Size> full_img_sizes(num_images);
		double seam_work_aspect = 1;

		#pragma omp parallel for
		for (int i = 0; i < num_images; ++i)
		{
			full_img[i] = orig_images[i];
			full_img_sizes[i] = full_img[i].size();

			if (full_img[i].empty())
			{
				// LOGLN("Can't open image " << img_names[i]);
				cout << "Cannot open images." << endl;
			}
			if (work_megapix < 0)
			{
				img[i] = full_img[i];
				work_scale = 1;
				is_work_scale_set = true;
			}
			else
			{
				if (!is_work_scale_set)
				{
					work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img[i].size().area()));
					is_work_scale_set = true;
				}
				resize(full_img[i], img[i], Size(), work_scale, work_scale);
			}
			if (!is_seam_scale_set)
			{
				seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img[i].size().area()));
				seam_work_aspect = seam_scale / work_scale;
				is_seam_scale_set = true;
			}

			(*finder)(img[i], features[i]);
			features[i].img_idx = i;
			LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());

			resize(full_img[i], img[i], Size(), seam_scale, seam_scale);
			images[i] = img[i].clone();
		}

		finder->collectGarbage();

		LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

		LOG("Pairwise matching");
#if ENABLE_LOG
		t = getTickCount();
#endif
		vector<MatchesInfo> pairwise_matches;
		BestOf2NearestMatcher matcher(try_gpu, match_conf);
		Mat matchMask(features.size(), features.size(), CV_8U, Scalar(0));
		for (int i = 0; i < num_images - 1; ++i)
		{
			matchMask.at<char>(i, i + 1) = 1;
		}
		matcher(features, pairwise_matches, matchMask);
		matcher.collectGarbage();

		LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

		// Check if we should save matches graph
		if (save_graph)
		{
			LOGLN("Saving matches graph...");
			ofstream f(save_graph_to.c_str());
			// f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
		}

		// Leave only images we are sure are from the same panorama
		vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
		vector<Mat> img_subset, untouched_images;
		vector<Size> full_img_sizes_subset;
		for (size_t i = 0; i < indices.size(); ++i)
		{
			untouched_images.push_back(orig_images[indices[i]]);
			img_subset.push_back(images[indices[i]]);
			full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
		}
		cout << img_subset.size() << " images left" << endl;

		images = img_subset;
		full_img_sizes = full_img_sizes_subset;

		// Check if we still have enough images
		num_images = static_cast<int>(images.size());
		if (num_images < 2)
		{
			LOGLN("Need more images");
			cout << "Need more images." << endl;
			return false;
		}

		HomographyBasedEstimator estimator;
		vector<CameraParams> cameras;
		estimator(features, pairwise_matches, cameras);

		for (size_t i = 0; i < cameras.size(); ++i)
		{
			Mat R;
			cameras[i].R.convertTo(R, CV_32F);
			cameras[i].R = R;
			LOGLN("Initial intrinsics #" << indices[i] + 1 << ":\n" << cameras[i].K());
		}

		Ptr<detail::BundleAdjusterBase> adjuster;
		if (ba_cost_func == "reproj") adjuster = new detail::BundleAdjusterReproj();
		else if (ba_cost_func == "ray") adjuster = new detail::BundleAdjusterRay();
		else
		{
			cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
			return false;
		}
		adjuster->setConfThresh(conf_thresh);
		Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
		if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
		if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
		if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
		if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
		if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
		adjuster->setRefinementMask(refine_mask);
		(*adjuster)(features, pairwise_matches, cameras);

		// Find median focal length

		vector<double> focals;
		for (size_t i = 0; i < cameras.size(); ++i)
		{
			LOGLN("Camera #" << indices[i] + 1 << ":\n" << cameras[i].K());
			focals.push_back(cameras[i].focal);
		}

		sort(focals.begin(), focals.end());
		float warped_image_scale;
		if (focals.size() % 2 == 1)
			warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
		else
			warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

		if (do_wave_correct)
		{
			vector<Mat> rmats;
			for (size_t i = 0; i < cameras.size(); ++i)
				rmats.push_back(cameras[i].R);
			waveCorrect(rmats, wave_correct);
			for (size_t i = 0; i < cameras.size(); ++i)
				cameras[i].R = rmats[i];
		}

		LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
		t = getTickCount();
#endif

		vector<Point> corners(num_images);
		vector<Mat> masks_warped(num_images);
		vector<Mat> images_warped(num_images);
		vector<Size> sizes(num_images);
		vector<Mat> masks(num_images);

		// Preapre images masks
		for (int i = 0; i < num_images; ++i)
		{
			masks[i].create(images[i].size(), CV_8U);
			masks[i].setTo(Scalar::all(255));
		}

		// Warp images and their masks

		Ptr<WarperCreator> warper_creator;
#if defined(HAVE_OPENCV_GPU)
		if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
		{
			if (warp_type == "plane") warper_creator = new cv::PlaneWarperGpu();
			else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarperGpu();
			else if (warp_type == "spherical") warper_creator = new cv::SphericalWarperGpu();
		}
		else
#endif
		{
			if (warp_type == "plane") warper_creator = new cv::PlaneWarper();
			else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarper();
			else if (warp_type == "spherical") warper_creator = new cv::SphericalWarper();
			else if (warp_type == "fisheye") warper_creator = new cv::FisheyeWarper();
			else if (warp_type == "stereographic") warper_creator = new cv::StereographicWarper();
			else if (warp_type == "compressedPlaneA2B1") warper_creator = new cv::CompressedRectilinearWarper(2, 1);
			else if (warp_type == "compressedPlaneA1.5B1") warper_creator = new cv::CompressedRectilinearWarper(1.5, 1);
			else if (warp_type == "compressedPlanePortraitA2B1") warper_creator = new cv::CompressedRectilinearPortraitWarper(2, 1);
			else if (warp_type == "compressedPlanePortraitA1.5B1") warper_creator = new cv::CompressedRectilinearPortraitWarper(1.5, 1);
			else if (warp_type == "paniniA2B1") warper_creator = new cv::PaniniWarper(2, 1);
			else if (warp_type == "paniniA1.5B1") warper_creator = new cv::PaniniWarper(1.5, 1);
			else if (warp_type == "paniniPortraitA2B1") warper_creator = new cv::PaniniPortraitWarper(2, 1);
			else if (warp_type == "paniniPortraitA1.5B1") warper_creator = new cv::PaniniPortraitWarper(1.5, 1);
			else if (warp_type == "mercator") warper_creator = new cv::MercatorWarper();
			else if (warp_type == "transverseMercator") warper_creator = new cv::TransverseMercatorWarper();
		}

		if (warper_creator.empty())
		{
			cout << "Can't create the following warper '" << warp_type << "'\n";
			return false;
		}

		Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
		vector<Mat> images_warped_f(num_images);

		for (int i = 0; i < num_images; ++i)
		{
			Mat_<float> K;
			cameras[i].K().convertTo(K, CV_32F);
			float swa = (float)seam_work_aspect;
			K(0, 0) *= swa; K(0, 2) *= swa;
			K(1, 1) *= swa; K(1, 2) *= swa;

			corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
			sizes[i] = images_warped[i].size();
			cout << "warp corners and sizes" << corners[i] << " " << sizes[i] << endl;

			warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
			images_warped[i].convertTo(images_warped_f[i], CV_32F);
		}



		LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

		Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
		compensator->feed(corners, images_warped, masks_warped);

		Ptr<SeamFinder> seam_finder;
		if (seam_find_type == "no")
			seam_finder = new detail::NoSeamFinder();
		else if (seam_find_type == "voronoi")
			seam_finder = new detail::VoronoiSeamFinder();
		else if (seam_find_type == "gc_color")
		{
#if defined(HAVE_OPENCV_GPU)
			if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
				seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR);
			else
#endif
				seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
		}
		else if (seam_find_type == "gc_colorgrad")
		{
#if defined(HAVE_OPENCV_GPU)
			if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
				seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR_GRAD);
			else
#endif
				seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		}
		else if (seam_find_type == "dp_color")
			seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
		else if (seam_find_type == "dp_colorgrad")
			seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);
		if (seam_finder.empty())
		{
			cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
			return false;
		}

		seam_finder->find(images_warped_f, corners, masks_warped);

		// Release unused memory
		images.clear();
		images_warped.clear();
		images_warped_f.clear();
		masks.clear();

		LOGLN("Compositing...");
#if ENABLE_LOG
		t = getTickCount();
#endif

		Mat img_warped, img_warped_s;
		Mat dilated_mask, seam_mask, mask, mask_warped;
		Ptr<Blender> blender;
		//double compose_seam_aspect = 1;
		double compose_work_aspect = 1;

		//#pragma omp parallel for
		for (int img_idx = 0; img_idx < num_images; ++img_idx)
		{
			LOGLN("Compositing image #" << indices[img_idx] + 1);

			// Read image and resize it if necessary
			if (!is_compose_scale_set)
			{
				if (compose_megapix > 0)
					compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img[img_idx].size().area()));
				is_compose_scale_set = true;

				// Compute relative scales
				//compose_seam_aspect = compose_scale / seam_scale;
				compose_work_aspect = compose_scale / work_scale;

				// Update warped image scale
				warped_image_scale *= static_cast<float>(compose_work_aspect);
				warper = warper_creator->create(warped_image_scale);

				// Update corners and sizes
				for (int i = 0; i < num_images; ++i)
				{
					cout << "prev corners and sizes" << corners[i] << " " << sizes[i] << endl;
					// Update intrinsics
					cameras[i].focal *= compose_work_aspect;
					cameras[i].ppx *= compose_work_aspect;
					cameras[i].ppy *= compose_work_aspect;

					// Update corner and size
					Size sz = full_img_sizes[i];

					if (std::abs(compose_scale - 1) > 1e-1)
					{
						sz.width = cvRound(full_img_sizes[i].width * compose_scale);
						sz.height = cvRound(full_img_sizes[i].height * compose_scale);
					}

					Mat K;
					cameras[i].K().convertTo(K, CV_32F);
					Rect roi = warper->warpRoi(sz, K, cameras[i].R);
					corners[i] = roi.tl();
					sizes[i] = roi.size();
					cout << "post corners and sizes" << corners[i] << " " << sizes[i] << endl;
				}
			}
			if (abs(compose_scale - 1) > 1e-1)
				resize(full_img[img_idx], img[img_idx], Size(), compose_scale, compose_scale);
			else
				img[img_idx] = full_img[img_idx];
			full_img[img_idx].release();
			Size img_size = img[img_idx].size();

			cout << img_size << "img_size" << endl;

			Mat K;
			cameras[img_idx].K().convertTo(K, CV_32F);

			// Warp the current image
			warper->warp(img[img_idx], K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

			// Warp the current image mask
			mask.create(img_size, CV_8U);
			mask.setTo(Scalar::all(255));
			warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

			// Compensate exposure
			compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

			img_warped.convertTo(img_warped_s, CV_16S);
			img_warped.release();
			img[img_idx].release();
			mask.release();

			dilate(masks_warped[img_idx], dilated_mask, Mat());
			resize(dilated_mask, seam_mask, mask_warped.size());
			mask_warped = seam_mask & mask_warped;

			if (blender.empty())
			{
				blender = Blender::createDefault(blend_type, try_gpu);
				Size dst_sz = resultRoi(corners, sizes).size();
				float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
				if (blend_width < 1.f) {
					blender = Blender::createDefault(Blender::NO, try_gpu);
				}
				else if (blend_type == Blender::MULTI_BAND)
				{
					MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
					mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
					LOGLN("Multi-band blender, number of bands: " << mb->numBands());
				}
				else if (blend_type == Blender::FEATHER)
				{
					FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
					fb->setSharpness(1.f / blend_width);
					LOGLN("Feather blender, sharpness: " << fb->sharpness());
				}
				cout << "about to prepare" << endl;
				blender->prepare(corners, sizes);
				cout << "prepared" << endl;
			}

			// Blend the current image
			blender->feed(img_warped_s, mask_warped, corners[img_idx]);

			cout << "blended" << endl;
		}

		Mat result;
		Mat result_mask;
		blender->blend(result, result_mask);

		LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

		cout << "size " << result.size() << " " << result.size[0] << " " << result.cols << endl;

		if (result.rows > (int) 1 || result.cols > (int) 1) {
			imwrite(result_name, result);
			LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
			return true;
		}
		else {
			std::cout << "empty image" << std::endl;
			return false;
		}
	} catch (cv::Exception &e) {
		const char* err_msg = e.what();
		cout << "error: " << err_msg << endl;
		return false;
	}
}

bool stitch2(vector<Mat> imgs) {
	Stitcher stitcher = Stitcher::createDefault(false);
	stitcher.setWarper(new cv::PlaneWarper());
	stitcher.setFeaturesFinder(new detail::SurfFeaturesFinder(300,3,4,3,4));
	stitcher.setRegistrationResol(0.08);
	stitcher.setSeamEstimationResol(0.08);
	stitcher.setCompositingResol(-1);
	stitcher.setPanoConfidenceThresh(1);
	stitcher.setWaveCorrection(true);
	stitcher.setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);
	stitcher.setFeaturesMatcher(new detail::BestOf2NearestMatcher(false, 0.3));
	stitcher.setBundleAdjuster(new detail::BundleAdjusterRay());
	Stitcher::Status status = Stitcher::ERR_NEED_MORE_IMGS;

	try {
		if (imgs.size() == 0) return true;
		if (imgs.size() == 1) {
			imwrite(result_name, imgs[0]);
			return true;
		}
		Mat pano;
		status = stitcher.stitch(imgs, pano);
		//imshow("pano", pano);
		//waitKey(0);
		imwrite(result_name, pano);
		return status;
	}
	catch(cv::Exception &e) {
		const char* err_msg = e.what();
		cout << "error: " << err_msg << endl;
		return false;
	}
}

void displayVideo(VideoCapture cap) {
	cout << "here" << endl;
	while (1)
	{
		Mat frame;
		cap >> frame;
		imshow("stream", frame);
		waitKey(1);
		
		if ( frame.empty()) {
			cout << "frame empty" << endl;
			break; // end of video stream
		}
	}
}

int main(int argc, char const *argv[])
{
	int64 app_start_time = getTickCount();

	if (argc != 4) {
		cout << "Usage: 	Name of video file	Result name with extension	Number of frames between stitches." << endl;
	}	

	namedWindow("stream", WINDOW_NORMAL);

	VideoCapture cap;

	if (!cap.open(argv[1])) {
		cout << "Failed to open video." << endl;
		return -1;
	}

	std::future<void> fut = std::async(displayVideo, cap);

	int frameSkip = atoi(argv[3]);

	cout << frameSkip << endl;

	vector<Mat> images;
	int curr_frame = 0;
	double frame_count = cap.get(CV_CAP_PROP_FRAME_COUNT);
	double fps = cap.get(CV_CAP_PROP_FPS);

	// Extract each video frame:
	while (1)
	{
		Mat frame;
		cap >> frame;
		
		if ( frame.empty()) {
			cout << "frame empty" << endl;
			imwrite(argv[2],  (imread("temp.jpg", CV_LOAD_IMAGE_COLOR)));
			break; // end of video stream
		}

		if ((curr_frame % frameSkip == 0)/* || (curr_frame == (int) frame_count - 1)*/) {
			cout << curr_frame << " " << frame.size() << endl;
			images.push_back(frame);
			bool success = stitch(images);
			// bool success = stitch2(images);

			images.clear();
			Mat temp = imread("temp.jpg", CV_LOAD_IMAGE_COLOR);

			if (!temp.empty())
				images.push_back(temp);
			else
				images.push_back(frame);
			cout << success <<  " stitch done" << endl << endl;
			imshow("stream", temp);
			waitKey(1000);
		}

		frame.release();
		curr_frame++;

		if ( curr_frame == (int) frame_count) {
			imwrite(argv[2],  (imread("temp.jpg", CV_LOAD_IMAGE_COLOR)));
			break; // end of video stream
		}
	}

	images.clear();
	cout << "Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec" << endl;
	destroyAllWindows();
	return 1;
}