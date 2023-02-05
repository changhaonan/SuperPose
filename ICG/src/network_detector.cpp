#include <icg/network_detector.h>

namespace icg
{

    NetworkDetector::NetworkDetector(const std::string &name,
                                     const std::shared_ptr<Body> &body_ptr,
                                     const Transform3fA &body2world_pose,
                                     const std::shared_ptr<ColorCamera>& color_camera_ptr)
        : Detector{name}, body_ptr_{body_ptr}, body2world_pose_{body2world_pose}, color_camera_ptr_(color_camera_ptr) {}

    NetworkDetector::NetworkDetector(const std::string &name,
                                     const std::filesystem::path &metafile_path,
                                     const std::shared_ptr<icg::Body> &body_ptr,
                                     const std::shared_ptr<ColorCamera>& color_camera_ptr)
        : Detector{name, metafile_path}, body_ptr_{body_ptr}, color_camera_ptr_(color_camera_ptr) {};

    NetworkDetector::~NetworkDetector()
    {
        close(socket_fd_);
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
        // Build connection to port
        if (!ConnectToServer())
            return false;

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
            std::cerr << "Set up static detector " << name_ << " first" << std::endl;
            return false;
        }
        body_ptr_->set_body2world_pose(body2world_pose_);

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
        if (!ReadRequiredValueFromYaml(fs, "body2world_pose", &body2world_pose_) ||
            !ReadRequiredValueFromYaml(fs, "port", &port_))
        {
            std::cerr << "Could not read all required body parameters from "
                      << metafile_path_ << std::endl;
            return false;
        }
        fs.release();
        return true;
    }

    bool NetworkDetector::ConnectToServer()
    {
        // Create socket
        socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd_ < 0)
        {
            std::cerr << "Could not create socket" << std::endl;
            return false;
        }

        // Set server address
        server_address_.sin_family = AF_INET;
        server_address_.sin_port = htons(port_);
        server_address_.sin_addr.s_addr = INADDR_ANY;

        // Connect to server
        if (connect(socket_fd_, (struct sockaddr *)&server_address_, sizeof(server_address_)) < 0)
        {
            std::cerr << "Could not connect to server" << std::endl;
            return false;
        }
        return true;
    }

    bool NetworkDetector::ReadPoseFromSocket()
    {   
        // Send image to server
        cv::Mat image = color_camera_ptr_->image();
        std::vector<uchar> image_buffer;
        cv::imencode(".png", image, image_buffer);
        int image_size = image_buffer.size();
        // send(socket_fd_, &image_size, sizeof(int), 0);
        send(socket_fd_, image_buffer.data(), image_size, 0);

        // Read pose from socket
        char buffer[1024] = {0};
        int valread = read(socket_fd_, buffer, 1024);
        if (valread < 0)
        {
            std::cerr << "Could not read from socket" << std::endl;
            return false;
        }
        else
        {
            // Recover pose from buffer
            std::string pose_string(buffer);
            std::stringstream ss(pose_string);
            
            // Parse pose from string
            Eigen::Matrix4f pose;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    ss >> pose(i, j);
            std::cout << "Received pose: " << std::endl
                      << pose << std::endl;
            // Eigen matrix to Affine3f
            Eigen::Affine3f pose_affine(pose);
            body2world_pose_ = pose_affine.matrix();
            body_ptr_->set_body2world_pose(body2world_pose_);
        }
        return true;
    }

} // namespace icg
