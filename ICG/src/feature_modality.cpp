#include <icg/feature_modality.h>

namespace icg
{
    FeatureModality::FeatureModality(const std::string &name, const std::shared_ptr<Body> &body_ptr,
                                     const std::shared_ptr<ColorCamera> &color_camera_ptr,
                                     const std::shared_ptr<DepthCamera> &depth_camera_ptr,
                                     const std::shared_ptr<FeatureModel> &feature_model_ptr)
        : Modality(name, body_ptr), color_camera_ptr_(color_camera_ptr), depth_camera_ptr_(depth_camera_ptr), feature_model_ptr_(feature_model_ptr)
    {
        if (depth_camera_ptr_ == nullptr)
        {
            std::cerr << "Depth camera is disabled" << std::endl;
            depth_enabled_ = false;
        }
        else
        {
            depth_enabled_ = true;
        }
    }

    FeatureModality::FeatureModality(
        const std::string &name, const std::filesystem::path &metafile_path,
        const std::shared_ptr<Body> &body_ptr,
        const std::shared_ptr<ColorCamera> &color_camera_ptr,
        const std::shared_ptr<DepthCamera> &depth_camera_ptr,
        const std::shared_ptr<FeatureModel> &feature_model_ptr)
        : Modality(name, metafile_path, body_ptr), color_camera_ptr_(color_camera_ptr), depth_camera_ptr_(depth_camera_ptr), feature_model_ptr_(feature_model_ptr)
    {
        if (depth_camera_ptr_ == nullptr)
        {
            std::cerr << "Depth camera is disabled" << std::endl;
            depth_enabled_ = false;
        }
        else
        {
            depth_enabled_ = true;
        }
    }

    bool FeatureModality::SetUp()
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
        if (!feature_model_ptr_->set_up())
        {
            std::cerr << "Feature model " << feature_model_ptr_->name() << " was not set up"
                      << std::endl;
            return false;
        }
        if (depth_enabled_ && !depth_camera_ptr_->set_up())
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

        // Set up the feature manager (Shared from feature model)
        feature_manager_ptr_ = feature_model_ptr_->feature_manager_ptr();

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

        PrecalculatePoseVariables();
        PrecalculateIterationDependentVariables(corr_iteration);

        // Compute current ROI first

        // Create new frame object
        cv::Mat color_image = color_camera_ptr_->image();
        if (depth_enabled_)
        {
            cv::Mat depth_image = depth_camera_ptr_->image();
            current_frame_ptr_ = FeatureModel::WrapFrame(color_image, depth_image);
        }
        else
        {
            cv::Mat depth_image;
            current_frame_ptr_ = FeatureModel::WrapFrame(color_image, depth_image);
        }
        feature_manager_ptr_->detectFeature(current_frame_ptr_, 0);

        // Search closest template view
        const FeatureModel::View *view;
        feature_model_ptr_->GetClosestView(body2camera_pose_, &view);

        // Compute correspondences
        std::vector<cv::DMatch> matches;
        MatchFeatures(view->feature_descriptor, current_frame_ptr_->_feat_des, matches);

        std::cout << "Found " << matches.size() << " matches.." << std::endl;
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

    void FeatureModality::PrecalculateCameraVariables()
    {
        // We majorly use the color camera intrinsics
        fu_ = color_camera_ptr_->intrinsics().fu;
        fv_ = color_camera_ptr_->intrinsics().fv;
        ppu_ = color_camera_ptr_->intrinsics().ppu;
        ppv_ = color_camera_ptr_->intrinsics().ppv;
        image_width_minus_1_ = color_camera_ptr_->intrinsics().width - 1;
        image_height_minus_1_ = color_camera_ptr_->intrinsics().height - 1;
    }

    bool FeatureModality::PrecalculateModelVariables()
    {
        float stride = feature_model_ptr_->stride_depth_offset();
        float max_radius = feature_model_ptr_->max_radius_depth_offset();
        return true;
    }

    void FeatureModality::PrecalculateRendererVariables()
    {
    }

    void FeatureModality::PrecalculatePoseVariables()
    {
        body2camera_pose_ =
            color_camera_ptr_->world2camera_pose() * body_ptr_->body2world_pose();
        camera2body_pose_ = body2camera_pose_.inverse();
        body2camera_rotation_ = body2camera_pose_.rotation().matrix();
    }

    void FeatureModality::PrecalculateIterationDependentVariables(
        int corr_iteration)
    {
    }

    void FeatureModality::VisualizePointsFeatureImage(const std::string &title, int save_idx) const
    {
        // Visualize the current feature object
        cv::Mat visualization_image = current_frame_ptr_->_color.clone();
        for (auto &kpts : current_frame_ptr_->_keypts)
        {
            cv::circle(visualization_image, kpts.pt, 2, cv::Scalar(0, 255, 0), 2);
        }

        // Visualize the current feature view

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

    // Helper methods for Correspodence
    bool FeatureModality::MatchFeatures(
        const cv::Mat &view_descriptor, const cv::Mat &frame_descriptor, std::vector<cv::DMatch> &matches, float ratio_thresh)
    {
        // Create cv::Mat from std::vector<float>
        cv::Mat descriptor_query_mat; // Query descriptor
        cv::Mat descriptor_train_mat;

        view_descriptor.convertTo(descriptor_query_mat, CV_32F);
        frame_descriptor.convertTo(descriptor_train_mat, CV_32F);

        // Match
        cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(descriptor_query_mat, descriptor_train_mat, knn_matches, 2);

        // Filter matches using the Lowe's ratio test
        std::vector<std::pair<int, int>> good_matches_int2;
        for (auto j = 0; j < knn_matches.size(); j++)
        {
            if (knn_matches[j].size() >= 2)
            {
                if (knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance)
                {
                    matches.push_back(knn_matches[j][0]);
                }
            }
        }

        return true;
    }

    // Helper functions
    bool FeatureModality::IsSetup() const
    {
        if (!set_up_)
        {
            std::cerr << "Set up feature modality " << name_ << " first" << std::endl;
            return false;
        }
        return true;
    }
}
