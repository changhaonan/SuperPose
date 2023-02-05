#include <icg/network_detector.h>

namespace icg
{

    NetworkDetector::NetworkDetector(const std::string &name,
                                     const std::shared_ptr<Body> &body_ptr,
                                     const Transform3fA &body2world_pose)
        : Detector{name}, body_ptr_{body_ptr}, body2world_pose_{body2world_pose} {}

    NetworkDetector::NetworkDetector(const std::string &name,
                                     const std::filesystem::path &metafile_path,
                                     const std::shared_ptr<icg::Body> &body_ptr)
        : Detector{name, metafile_path}, body_ptr_{body_ptr} {};

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
        if (!ReadRequiredValueFromYaml(fs, "body2world_pose", &body2world_pose_))
        {
            std::cerr << "Could not read all required body parameters from "
                      << metafile_path_ << std::endl;
            return false;
        }
        fs.release();
        return true;
    }

} // namespace icg
