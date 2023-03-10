#pragma once
#include <icg/body.h>
#include <icg/common.h>
#include <icg/detector.h>

#include <filesystem/filesystem.h>
#include <Eigen/Geometry>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cppzmq/zmq.hpp>
#include <cppzmq/zmq_addon.hpp>

namespace icg
{
    /**
     * \brief \ref Detector that assigns a listens pose from a network socket.
     *
     * @param body_ptr referenced \ref Body object for which the pose is set.
     * @param body2world_pose pose that is assigned to the \ref Body object.
     */
    class NetworkDetector : public Detector
    {
    public:
        // Constructor and setup method
        NetworkDetector(const std::string &name, const std::shared_ptr<Body> &body_ptr,
                        const Transform3fA &body2world_pose,
                        const std::shared_ptr<ColorCamera>& color_camera_ptr);
        NetworkDetector(const std::string &name,
                        const std::filesystem::path &metafile_path,
                        const std::shared_ptr<icg::Body> &body_ptr,
                        const std::shared_ptr<ColorCamera>& color_camera_ptr);
        ~NetworkDetector();
        bool SetUp() override;

        // Setters
        void set_body_ptr(const std::shared_ptr<Body> &body_ptr);
        void set_body2world_pose(const Transform3fA &body2world_pose);

        // Main methods
        bool DetectBody(int iteration) override;

        // Getters
        const std::shared_ptr<Body> &body_ptr() const;
        const Transform3fA &body2world_pose() const;
        std::vector<std::shared_ptr<Body>> body_ptrs() const override;

    private:
        bool LoadMetaData();
        bool ReadPoseFromSocket();
        std::shared_ptr<Body> body_ptr_{};
        Transform3fA body2world_pose_{};
        int port_;
        zmq::context_t context_;
        zmq::socket_t socket_;
        std::shared_ptr<ColorCamera> color_camera_ptr_;
    };

}