#include <icg/network_detector.h>

namespace icg
{

    NetworkDetector::NetworkDetector(const std::string &name,
                                     const std::shared_ptr<Body> &body_ptr,
                                     const Transform3fA &body2world_pose,
                                     const std::shared_ptr<ColorCamera> &color_camera_ptr)
        : Detector{name}, body_ptr_{body_ptr}, body2world_pose_{body2world_pose}, color_camera_ptr_(color_camera_ptr), socket_(context_, ZMQ_REQ) {}

    NetworkDetector::NetworkDetector(const std::string &name,
                                     const std::filesystem::path &metafile_path,
                                     const std::shared_ptr<icg::Body> &body_ptr,
                                     const std::shared_ptr<ColorCamera> &color_camera_ptr)
        : Detector{name, metafile_path}, body_ptr_{body_ptr}, color_camera_ptr_(color_camera_ptr), socket_(context_, ZMQ_REQ){};

    NetworkDetector::~NetworkDetector()
    {
    }

    bool NetworkDetector::SetUp()
    {
        set_up_ = false;
        if (!metafile_path_.empty())
            if (!LoadMetaData())
                return false;

        // Check if all required objects are set up
        if (!body_ptr_->set_up())
        {
            std::cerr << "Body " << body_ptr_->name() << " was not set up" << std::endl;
            return false;
        }
        // Set up zmq socket
        socket_.connect("tcp://0.0.0.0:" + std::to_string(port_));
        std::cout << "Connected to port " << port_ << "..." << std::endl;

        set_up_ = true;
        return true;
    }

    void NetworkDetector::set_body_ptr(const std::shared_ptr<Body> &body_ptr)
    {
        body_ptr_ = body_ptr;
        set_up_ = false;
    }

    void NetworkDetector::set_body2world_pose(const Transform3fA &body2world_pose)
    {
        body2world_pose_ = body2world_pose;
    }

    bool NetworkDetector::DetectBody()
    {
        if (!set_up_)
        {
            std::cerr << "Set up network detector " << name_ << " first" << std::endl;
            return false;
        }
        ReadPoseFromSocket();
        return true;
    }

    const std::shared_ptr<Body> &NetworkDetector::body_ptr() const
    {
        return body_ptr_;
    }

    const Transform3fA &NetworkDetector::body2world_pose() const
    {
        return body2world_pose_;
    }

    std::vector<std::shared_ptr<Body>> NetworkDetector::body_ptrs() const
    {
        return {body_ptr_};
    }

    bool NetworkDetector::LoadMetaData()
    {
        // Open file storage from yaml
        cv::FileStorage fs;
        if (!OpenYamlFileStorage(metafile_path_, &fs))
            return false;

        // Read parameters from yaml
        if (!ReadRequiredValueFromYaml(fs, "port", &port_))
        {
            std::cerr << "Could not read all required body parameters from "
                      << metafile_path_ << std::endl;
            return false;
        }
        fs.release();
        return true;
    }

    bool NetworkDetector::ReadPoseFromSocket()
    {
        // Send image to server using zmq
        cv::Mat image = color_camera_ptr_->image();
        {
            // Send image size first
            zmq::message_t msg(2 * sizeof(int));
            std::vector<int> wh = {image.cols, image.rows};
            std::memcpy(msg.data(), wh.data(), 2 * sizeof(int));
            socket_.send(msg, ZMQ_SNDMORE);
        }
        {
            cv::Mat flat = image.reshape(1, image.total() * image.channels());
            std::vector<unsigned char> vec = image.isContinuous() ? flat : flat.clone();
            zmq::message_t msg(vec.size() * sizeof(unsigned char));
            std::memcpy(msg.data(), vec.data(), vec.size() * sizeof(unsigned char));
            socket_.send(msg, 0);
        }

        std::cout << "[pose detector]: waiting for reply" << std::endl;
        std::vector<zmq::message_t> recv_msgs;
        zmq::recv_multipart(socket_, std::back_inserter(recv_msgs));
        std::cout << "[pose detector]: got reply" << std::endl;

        // Read & parse pose from socket
        Eigen::Matrix4f pose;
        std::memcpy(pose.data(), recv_msgs[0].data(), sizeof(Eigen::Matrix4f));
        // Check pose
        if (pose(3, 3) == 0.f)
        {
            std::cerr << "Pose is not valid" << std::endl;
            return false;
        }
        // Eigen matrix to Affine3f
        Eigen::Affine3f pose_affine(pose);
        body2world_pose_ = pose_affine.matrix();
        body_ptr_->set_body2world_pose(body2world_pose_);

        return true;
    }

} // namespace icg
