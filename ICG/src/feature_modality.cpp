#include <icg/feature_modality.h>

namespace icg
{
    FeatureModality::FeatureModality(const std::string &name, const std::shared_ptr<Body> &body_ptr,
                                     const std::shared_ptr<ColorCamera> &color_camera_ptr,
                                     const std::shared_ptr<DepthCamera> &depth_camera_ptr,
                                     const std::shared_ptr<FeatureModel> &feature_model_ptr)
        : Modality(name, body_ptr), color_camera_ptr_(color_camera_ptr), depth_camera_ptr_(depth_camera_ptr), feature_model_ptr_(feature_model_ptr)
    {
    }

    FeatureModality::FeatureModality(
        const std::string &name, const std::filesystem::path &metafile_path,
        const std::shared_ptr<Body> &body_ptr,
        const std::shared_ptr<ColorCamera> &color_camera_ptr,
        const std::shared_ptr<DepthCamera> &depth_camera_ptr,
        const std::shared_ptr<FeatureModel> &feature_model_ptr)
        : Modality(name, metafile_path, body_ptr), color_camera_ptr_(color_camera_ptr), depth_camera_ptr_(depth_camera_ptr), feature_model_ptr_(feature_model_ptr)
    {
    }

    bool FeatureModality::SetUp()
    {
        set_up_ = false;
        if (!metafile_path_.empty())
            if (!LoadMetaData())
                return false;

        // Set up the feature manager
        // Open the yaml file
        std::filesystem::path feature_manager_path(feature_config_file_);
        if (!std::filesystem::exists(feature_manager_path))
        {
            std::cerr << "Feature manager file " << feature_manager_path << " does not exist"
                      << std::endl;
            return false;
        }
        YAML::Node feature_manager_config = YAML::LoadFile(feature_manager_path.string());
        auto feature_manager_config_ptr = std::make_shared<YAML::Node>(feature_manager_config);
        feature_manager_ptr_ = std::make_shared<NetworkFeature>(feature_manager_config_ptr);

        // Check if all required objects are set up
        if (!body_ptr_->set_up())
        {
            std::cerr << "Body " << body_ptr_->name() << " was not set up" << std::endl;
            return false;
        }
        if (!feature_model_ptr_->set_up())
        {
            std::cerr << "Feature model " << feature_model_ptr_->name() << " was not set up"
                      << std::endl;
            return false;
        }
        if (!depth_camera_ptr_->set_up())
        {
            std::cerr << "Depth camera " << depth_camera_ptr_->name()
                      << " was not set up" << std::endl;
            return false;
        }
        if (!color_camera_ptr_->set_up())
        {
            std::cerr << "Color camera " << color_camera_ptr_->name()
                      << " was not set up" << std::endl;
            return false;
        }

        set_up_ = true;
        return true;
    }

    bool FeatureModality::StartModality(int iteration, int corr_iteration)
    {
        return IsSetup();
    }

    bool FeatureModality::CalculateCorrespondences(int iteration,
                                                   int corr_iteration)
    {
        if (!IsSetup())
            return false;
        // Create new frame object
        cv::Mat color_image = color_camera_ptr_->image();
        cv::Mat depth_image = depth_camera_ptr_->image();
        current_frame_ptr_ = WrapFrame(color_image, depth_image);
        feature_manager_ptr_->detectFeature(current_frame_ptr_, 0);

        return true;
    }

    bool FeatureModality::VisualizeCorrespondences(int save_idx)
    {
        if (!IsSetup())
            return false;

        if (visualize_correspondences_correspondence_)
            VisualizeCorrespondences("correspondences_correspondence", save_idx);
        return true;
    }

    std::shared_ptr<Frame> FeatureModality::WrapFrame(cv::Mat &color_image, cv::Mat &depth_image)
    {
        auto frame_ptr = std::make_shared<Frame>();
        frame_ptr->_color = color_image;
        frame_ptr->_depth = depth_image;

        // ROI
        Eigen::Vector4f roi;
        roi << 0, color_image.cols, 0, color_image.rows;
        frame_ptr->_roi = roi;
        return frame_ptr;
    }

    bool FeatureModality::LoadMetaData()
    {
        // Open file storage from yaml
        cv::FileStorage fs;
        if (!OpenYamlFileStorage(metafile_path_, &fs))
            return false;

        // Read required parameters from yaml
        if (!ReadRequiredValueFromYaml(fs, "port", &port_) ||
            !ReadRequiredValueFromYaml(fs, "config_path", &feature_config_file_))
            return false;

        // Read parameters from yaml for visualization
        ReadOptionalValueFromYaml(fs, "visualize_pose_result",
                                  &visualize_pose_result_);
        ReadOptionalValueFromYaml(fs, "visualize_gradient_optimization",
                                  &visualize_gradient_optimization_);
        ReadOptionalValueFromYaml(fs, "visualize_hessian_optimization",
                                  &visualize_hessian_optimization_);
        ReadOptionalValueFromYaml(fs, "visualize_correspondences_correspondence",
                                  &visualize_correspondences_correspondence_);
        ReadOptionalValueFromYaml(fs, "visualize_points_correspondence",
                                  &visualize_points_correspondence_);
        ReadOptionalValueFromYaml(fs,
                                  "visualize_points_depth_rendering_correspondence",
                                  &visualize_points_depth_rendering_correspondence_);
        ReadOptionalValueFromYaml(fs, "visualize_points_optimization",
                                  &visualize_points_optimization_);
        ReadOptionalValueFromYaml(fs, "visualize_points_result",
                                  &visualize_points_result_);
        ReadOptionalValueFromYaml(fs, "display_visualization",
                                  &display_visualization_);
        ReadOptionalValueFromYaml(fs, "save_visualizations", &save_visualizations_);
        ReadOptionalValueFromYaml(fs, "save_directory", &save_directory_);
        ReadOptionalValueFromYaml(fs, "save_image_type", &save_image_type_);

        return true;
    }

    void FeatureModality::VisualizePointsFeatureImage(const std::string &title, int save_idx) const
    {
        // Visualize the current feature object
        cv::Mat visualization_image = current_frame_ptr_->_color.clone();
        for (auto &kpts : current_frame_ptr_->_keypts)
        {
            cv::circle(visualization_image, kpts.pt, 2, cv::Scalar(0, 255, 0), 2);
        }
        ShowAndSaveImage(name_ + "_" + title, save_idx, visualization_image);
    }

    void FeatureModality::ShowAndSaveImage(const std::string &title, int save_idx,
                                           const cv::Mat &image) const
    {
        if (display_visualization_)
            cv::imshow(title, image);
        if (save_visualizations_)
        {
            std::filesystem::path path{
                save_directory_ /
                (title + "_" + std::to_string(save_idx) + "." + save_image_type_)};
            cv::imwrite(path.string(), image);
        }
    }

    bool FeatureModality::CalculateGradientAndHessian(int iteration,
                                                      int corr_iteration,
                                                      int opt_iteration)
    {
        if (!IsSetup())
            return false;
        gradient_.setZero();
        hessian_.setZero();

        return true;
    }

    bool FeatureModality::VisualizeOptimization(int save_idx)
    {
        if (!IsSetup())
            return false;

        if (visualize_points_optimization_)
        {
            // VisualizePointsDepthImage("depth_image_optimization", save_idx);
            // FIXME: To be implemented
        }
        if (visualize_gradient_optimization_)
        {
            VisualizeGradient();
        }
        if (visualize_hessian_optimization_)
        {
            VisualizeHessian();
        }
        return true;
    }

    bool FeatureModality::CalculateResults(int iteration) { return IsSetup(); }

    bool FeatureModality::VisualizeResults(int save_idx)
    {
        if (!IsSetup())
            return false;

        if (visualize_points_result_)
        {
            // Show feature and matching
            VisualizePointsFeatureImage("feature_result", save_idx);
        }
        if (visualize_pose_result_)
        {
            VisualizePose();
        }
        return true;
    }

    // Getters data
    const std::shared_ptr<ColorCamera> &FeatureModality::color_camera_ptr() const
    {
        return color_camera_ptr_;
    }

    const std::shared_ptr<FeatureModel> &FeatureModality::feature_model_ptr() const
    {
        return feature_model_ptr_;
    }

    std::shared_ptr<Model> FeatureModality::model_ptr() const
    {
        return feature_model_ptr_;
    }

    std::vector<std::shared_ptr<Camera>> FeatureModality::camera_ptrs() const
    {
        return {color_camera_ptr_};
    }

    std::vector<std::shared_ptr<Renderer>> FeatureModality::start_modality_renderer_ptrs() const
    {
        return {depth_renderer_ptr_};
    }

    std::vector<std::shared_ptr<Renderer>> FeatureModality::correspondence_renderer_ptrs() const
    {
        return {depth_renderer_ptr_};
    }

    std::vector<std::shared_ptr<Renderer>> FeatureModality::results_renderer_ptrs() const
    {
        return {depth_renderer_ptr_};
    }

    // Visualization method
    bool FeatureModality::VisualizeCorrespondences(const std::string &title, int save_idx)
    {
        return true;
    }

    // Helper functions
    bool FeatureModality::IsSetup() const
    {
        if (!set_up_)
        {
            std::cerr << "Set up depth modality " << name_ << " first" << std::endl;
            return false;
        }
        return true;
    }
}
