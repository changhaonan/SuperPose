/*
Authors: Bowen Wen
Contact: wenbowenxjtu@gmail.com
Created in 2021

Copyright (c) Rutgers University, 2021 All rights reserved.

Bowen Wen and Kostas Bekris. "BundleTrack: 6D Pose Tracking for Novel Objects
 without Instance or Category-Level 3D Models."
 In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2021.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Bowen Wen, Kostas Bekris, Rutgers University,
      nor the names of its contributors may be used to
      endorse or promote products derived from this software without
      specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <icg/frame.h>

zmq::context_t Frame::context;
zmq::socket_t Frame::socket;

MapPoint::MapPoint()
{
}

MapPoint::MapPoint(std::shared_ptr<Frame> frame, float u, float v)
{
    _img_pt[frame] = {u, v};
}

MapPoint::~MapPoint()
{
}

Frame::Frame()
{
}

Frame::Frame(const cv::Mat &color, const cv::Mat &depth, const cv::Mat &depth_raw, const cv::Mat &depth_sim, const Eigen::Vector4f &roi, const Eigen::Matrix4f &pose_in_model, int id, std::string id_str, const Eigen::Matrix3f &K, std::shared_ptr<YAML::Node> yml1)
{
    _status = OTHER;
    yml = yml1;
    _color = color;
    _vis = color.clone();
    _depth = depth;
    _depth_raw = depth_raw;
    _depth_sim = depth_sim;
    _H = color.rows;
    _W = color.cols;
    _id = id;
    _id_str = id_str;
    _pose_in_model = pose_in_model;
    _K = K;
    _pose_inited = false;
    _roi = roi;

    const int n_pixels = _H * _W;

    cv::cvtColor(_color, _gray, cv::COLOR_BGR2GRAY);
    processDepth();

    depthToCloudAndNormals();
}

Frame::~Frame()
{
}

void Frame::updateDepthCPU()
{
    std::cout << "updateDepthCPU implemented yet." << std::endl;
}

void Frame::updateDepthGPU()
{
    std::cout << "updateDepthGPU implemented yet." << std::endl;
}

void Frame::updateColorGPU()
{
    std::cout << "updateColorGPU implemented yet." << std::endl;
}

void Frame::updateNormalGPU()
{
    std::cout << "updateNormalGPU implemented yet." << std::endl;
}

void Frame::processDepth()
{
    std::cout << "processDepth implemented yet." << std::endl;
}

void Frame::depthToCloudAndNormals()
{
    std::cout << "depthToCloudAndNormals implemented yet." << std::endl;
}

void Frame::segmentationByMaskFile()
{
    const std::string data_dir = (*yml)["data_dir"].as<std::string>();
    int scene_id = -1;
    {
        std::regex pattern("scene_[0-9]");
        std::smatch what;
        if (std::regex_search(data_dir, what, pattern))
        {
            std::string result = what[0];
            boost::replace_all(result, "scene_", "");
            scene_id = std::stoi(result);
        }
    }

    std::string mask_file;
    if (data_dir.find("NOCS") != -1)
    {
        const std::string mask_dir = (*yml)["mask_dir"].as<std::string>();
        mask_file = mask_dir + "/" + _id_str + ".png";
    }
    else
    {
        mask_file = data_dir + "/masks/" + _id_str + ".png";
    }
    _fg_mask = cv::imread(mask_file, cv::IMREAD_UNCHANGED);
    if (_fg_mask.rows == 0)
    {
        printf("mask file open failed: %s\n", mask_file.c_str());
        exit(1);
    }

    if (data_dir.find("NOCS") != -1)
    {
        cv::Mat label;
        cv::connectedComponents(_fg_mask, label, 8);
        std::unordered_map<int, int> hist;
        for (int h = 0; h < _H; h++)
        {
            for (int w = 0; w < _W; w++)
            {
                if (_fg_mask.at<uchar>(h, w) == 0)
                    continue;
                hist[label.at<int>(h, w)]++;
            }
        }
        int max_num = 0;
        int max_id = 0;
        for (const auto &h : hist)
        {
            if (h.second > max_num)
            {
                max_num = h.second;
                max_id = h.first;
            }
        }

        if (max_num > 0)
        {
            std::vector<cv::Point2i> pts;
            for (int h = 0; h < _H; h++)
            {
                for (int w = 0; w < _W; w++)
                {
                    if (label.at<int>(h, w) == max_id && _fg_mask.at<uchar>(h, w) > 0)
                    {
                        pts.push_back({w, h});
                    }
                }
            }
            _fg_mask = cv::Mat::zeros(_H, _W, CV_8UC1);
            std::vector<cv::Point2i> hull;
            cv::convexHull(pts, hull);
            cv::fillConvexPoly(_fg_mask, hull, 1);
        }
        else
        {
            _fg_mask = cv::Mat::zeros(_H, _W, CV_8UC1);
        }
    }
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {5, 5});
    cv::dilate(_fg_mask, _fg_mask, kernel);

    invalidatePixelsByMask(_fg_mask);
}

void Frame::invalidatePixel(const int h, const int w)
{
    _color.at<cv::Vec3b>(h, w) = {0, 0, 0};
    _depth.at<float>(h, w) = 0;
    _depth_sim.at<float>(h, w) = 0;
    _depth_raw.at<float>(h, w) = 0;
    _gray.at<uchar>(h, w) = 0;
}

void Frame::invalidatePixelsByMask(const cv::Mat &fg_mask)
{
    assert(fg_mask.rows == _H && fg_mask.cols == _W);
    for (int h = 0; h < _H; h++)
    {
        for (int w = 0; w < _W; w++)
        {
            if (fg_mask.at<uchar>(h, w) == 0)
            {
                invalidatePixel(h, w);
            }
        }
    }
    updateColorGPU();
    updateDepthGPU();
    updateNormalGPU();

    _roi << 9999, 0, 9999, 0;
    for (int h = 0; h < _H; h++)
    {
        for (int w = 0; w < _W; w++)
        {
            if (fg_mask.at<uchar>(h, w) > 0)
            {
                _roi(0) = std::min(_roi(0), float(w));
                _roi(1) = std::max(_roi(1), float(w));
                _roi(2) = std::min(_roi(2), float(h));
                _roi(3) = std::max(_roi(3), float(h));
            }
        }
    }
    _fg_mask = fg_mask.clone();
}

bool Frame::operator==(const Frame &other)
{
    if (std::stoi(_id_str) == std::stoi(other._id_str))
        return true;
    return false;
}

bool Frame::operator<(const Frame &other)
{
    if (std::stoi(_id_str) < std::stoi(other._id_str))
        return true;
    return false;
}
