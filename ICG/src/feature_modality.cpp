#include <icg/feature_modality.h>

#define MATCHER_CLIENT_IMPLEMENTATION
#include <matcher_client.hpp>

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

        // Set up the matcher_clinet
        matcher_client_ptr_ = std::make_shared<pfh::MatcherClient>(port_, 400);
        if (!matcher_client_ptr_->SetUp())
        {
            std::cerr << "Failed to set up matcher client" << std::endl;
            return false;
        }

        PrecalculateCameraVariables();
        if (!PrecalculateModelVariables())
            return false;
        PrecalculateRendererVariables();

        // Set up the pnp_solver
        pnp_solver_ptr_ = std::make_shared<EPnPSolver>();
        if (!pnp_solver_ptr_->SetUp(color_camera_ptr_->metafile_path()))
        {
            std::cerr << "Failed to set up pnp solver" << std::endl;
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

        PrecalculatePoseVariables();
        PrecalculateIterationDependentVariables(corr_iteration);

        // Create new frame object
        cv::Mat color_image = color_camera_ptr_->image();
        // Compute current ROI first
        ComputeCurrentROI();

        // Search closest template view
        const FeatureModel::View *view;
        feature_model_ptr_->GetClosestView(body2camera_pose_, &view);
        // Compute relative rot_deg
        float relative_rot_deg = 0;
        feature_model_ptr_->GetRelativeRotDeg(body2camera_pose_, *view, relative_rot_deg);

        // Do matching
        std::vector<cv::KeyPoint> camera_kps;
        std::vector<cv::KeyPoint> body_kps;
        Eigen::Vector4f view_roi;
        view_roi << 0, view->texture_image.cols, 0, view->texture_image.rows;
        matcher_client_ptr_->Match(
            color_image, view->texture_image,
            current_roi_, view_roi,
            0, -relative_rot_deg,
            camera_kps, body_kps);

        // Construct data_points
        int num_matches = body_kps.size();
        for (int i = 0; i < std::min(n_points_, num_matches); ++i)
        {
            DataPoint data_point;
            CalculateBasicPointData(&data_point, *view, body_kps[i], camera_kps[i]);
            data_points_.push_back(data_point);
        }
        data_points_.resize(std::min(n_points_, num_matches));

        // Run Pnp
        int min_matches = 20;
        if (num_matches > min_matches)
        {
            RunPNP();
        }
        else
        {
            std::cout << "Not enough matches: " << num_matches << "/" << min_matches << std::endl;
        }
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

    bool FeatureModality::CalculateGradientAndHessian(int iteration,
                                                      int corr_iteration,
                                                      int opt_iteration)
    {
        if (!IsSetup())
            return false;

        PrecalculatePoseVariables();
        gradient_.setZero();
        hessian_.setZero();

        // for (auto &data_point : data_points_)
        // {
        //     // Calculate correspondence point coordinates in body frame
        //     Eigen::Vector3f correspondence_center_f_body =
        //         camera2body_pose_ * data_point.correspondence_center_f_camera;

        //     // Calculate intermediate variables
        //     float epsilon = data_point.normal_f_body.dot(data_point.center_f_body -
        //                                                  correspondence_center_f_body);
        //     Eigen::Vector3f correspondence_point_cross_normal =
        //         correspondence_center_f_body.cross(data_point.normal_f_body);

        //     // Calculate weight
        //     float correspondence_depth = data_point.correspondence_center_f_camera(2);
        //     float weight = 1.0f / (standard_deviation_ * correspondence_depth);
        //     float squared_weight = weight * weight;

        //     // Calculate weighted vectors
        //     Eigen::Vector3f weighted_correspondence_point_cross_normal =
        //         weight * correspondence_point_cross_normal;
        //     Eigen::Vector3f weighted_normal = weight * data_point.normal_f_body;

        //     // Calculate gradient
        //     gradient_.head<3>() -=
        //         (squared_weight * epsilon) * correspondence_point_cross_normal;
        //     gradient_.tail<3>() -=
        //         (squared_weight * epsilon) * data_point.normal_f_body;

        //     // Calculate hessian
        //     hessian_.topLeftCorner<3, 3>().triangularView<Eigen::Upper>() -=
        //         weighted_correspondence_point_cross_normal *
        //         weighted_correspondence_point_cross_normal.transpose();
        //     hessian_.topRightCorner<3, 3>() -=
        //         weighted_correspondence_point_cross_normal *
        //         weighted_normal.transpose();
        //     hessian_.bottomRightCorner<3, 3>().triangularView<Eigen::Upper>() -=
        //         weighted_normal * weighted_normal.transpose();
        // }
        // hessian_ = hessian_.selfadjointView<Eigen::Upper>();
        return true;
    }

    bool FeatureModality::LoadMetaData()
    {
        // Open file storage from yaml
        cv::FileStorage fs;
        if (!OpenYamlFileStorage(metafile_path_, &fs))
            return false;

        // Read required parameters from yaml
        if (!ReadRequiredValueFromYaml(fs, "port", &port_))
            return false;

        // Read Optional parameters from yaml
        ReadOptionalValueFromYaml(fs, "roi_margin", &roi_margin_);
        ReadOptionalValueFromYaml(fs, "n_points", &n_points_);

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
        // Do nothing
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

    bool FeatureModality::RunPNP()
    {
        // Get the 2D and 3D points
        std::vector<cv::Point2f> image_points;
        std::vector<cv::Point3f> object_points;
        for (auto i = 0; i < data_points_.size(); ++i)
        {
            image_points.push_back(data_points_[i].camera_uv);
            cv::Point3f object_point{data_points_[i].body_point.x(), data_points_[i].body_point.y(), data_points_[i].body_point.z()};
            object_points.push_back(object_point);
        }

        // Run PNP
        Eigen::Matrix3f rot_m;
        Eigen::Vector3f trans_m;
        rot_m = body2camera_rotation_;
        trans_m = body2camera_pose_.translation();

        if (!pnp_solver_ptr_->SolvePNP(object_points, image_points, rot_m, trans_m, true))
        {
            std::cout << "PNP failed." << std::endl;
            return false;
        }
        else
        {
            std::cout << "PNP succeeded." << std::endl;
            body_ptr_->set_body2world_pose(body2camera_pose_);
            return true;
        }
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

    int FeatureModality::n_points() const { return n_points_; }

    void FeatureModality::set_n_points(int n_points) { n_points_ = n_points; }

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

    void FeatureModality::ComputeCurrentROI()
    {
        // Compute the ROI by projecting max body diameter to the image plane and then expand the ROI by a margin
        float max_body_radius = body_ptr_->maximum_body_diameter() / 2.f;
        auto object_pose = body2camera_pose_.translation();
        float focal_length = std::max(fu_, fv_);
        float max_body_radius_in_image = max_body_radius * focal_length / object_pose.z();
        float center_x = object_pose.x() / object_pose.z() * fu_ + ppu_;
        float center_y = object_pose.y() / object_pose.z() * fv_ + ppv_;
        // Compute the ROI
        int roi_x = center_x - max_body_radius_in_image - roi_margin_;
        int roi_y = center_y - max_body_radius_in_image - roi_margin_;
        int roi_side = 2 * max_body_radius_in_image + 2 * roi_margin_;

        current_roi_ = Eigen::Vector4f(
            std::max(0, roi_x), std::min(image_width_minus_1_, roi_x + roi_side),
            std::max(0, roi_y), std::min(image_height_minus_1_, roi_y + roi_side));
    }

    void FeatureModality::CalculateBasicPointData(DataPoint *data_point, const FeatureModel::View &view,
                                                  const cv::KeyPoint &body_kps,
                                                  const cv::KeyPoint &camera_kps) const
    {
        // Depth
        cv::Point2i body_uv{int(body_kps.pt.x), int(body_kps.pt.y)};
        ushort body_depth{view.depth_image.at<ushort>(body_uv)};
        float body_depth_real = view.projection_term_a / (view.projection_term_b - float(body_depth));
        Eigen::Vector3f body_point{body_depth_real * (body_kps.pt.x - ppu_) / fu_,
                                   body_depth_real * (body_kps.pt.y - ppv_) / fv_,
                                   body_depth_real};

        // Normal
        auto normal_image_value{view.normal_image.at<cv::Vec4b>(body_uv)};
        Eigen::Vector3f body_normal{1.0f - float(normal_image_value[0]) / 127.5f,
                                    1.0f - float(normal_image_value[1]) / 127.5f,
                                    1.0f - float(normal_image_value[2]) / 127.5f};
        body_normal.normalize();

        // Set datapoint
        data_point->body_point = body_point;
        data_point->body_normal = body_normal;
        data_point->body_uv = body_uv;
        data_point->camera_uv = cv::Point2i(int(camera_kps.pt.x), (camera_kps.pt.y));
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

#undef MATCHER_CLIENT_IMPLEMENTATION