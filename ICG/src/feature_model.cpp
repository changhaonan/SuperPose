#include <icg/feature_model.h>

namespace icg
{

    FeatureModel::FeatureModel(const std::string &name,
                               const std::filesystem::path &metafile_path,
                               const std::shared_ptr<Body> &body_ptr)
        : Model(name, metafile_path, body_ptr)
    {
    }

    bool FeatureModel::SetUp()
    {
        set_up_ = false;
        if (!metafile_path_.empty())
            if (!LoadMetaData())
                return false;

        // Set up feature manager
        // Read yaml feature config first
        std::filesystem::path feature_manager_path(feature_config_path_);
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

        // Generate sparse model
        if (use_random_seed_ || !LoadModel())
        {
            if (!GenerateModel())
                return false;
            if (!SaveModel())
                return false;
        }

        set_up_ = true;
        return true;
    }

    bool FeatureModel::GetClosestView(const Transform3fA &body2camera_pose,
                                      const View **closest_view) const
    {
        if (!set_up_)
        {
            std::cerr << "Set up feature model " << name_ << " first" << std::endl;
            return false;
        }

        if (body2camera_pose.translation().norm() == 0.0f)
        {
            *closest_view = &views_[0];
            return true;
        }

        Eigen::Vector3f orientation{
            body2camera_pose.rotation().inverse() *
            body2camera_pose.translation().matrix().normalized()};

        float closest_dot = -1.0f;
        for (auto &view : views_)
        {
            float dot = orientation.dot(view.orientation);
            if (dot > closest_dot)
            {
                *closest_view = &view;
                closest_dot = dot;
            }
        }
        return true;
    }

    std::shared_ptr<NetworkFeature> FeatureModel::feature_manager_ptr() const
    {
        return feature_manager_ptr_;
    }

    bool FeatureModel::LoadMetaData()
    {
        // Open file storage from yaml
        cv::FileStorage fs;
        if (!OpenYamlFileStorage(metafile_path_, &fs))
            return false;

        // Read parameters from yaml
        if (!ReadRequiredValueFromYaml(fs, "model_path", &model_path_))
        {
            std::cerr << "Could not read all required body parameters from "
                      << metafile_path_ << std::endl;
            return false;
        }
        ReadOptionalValueFromYaml(fs, "sphere_radius", &sphere_radius_);
        ReadOptionalValueFromYaml(fs, "n_divides", &n_divides_);
        ReadOptionalValueFromYaml(fs, "n_points", &n_points_);
        ReadOptionalValueFromYaml(fs, "max_radius_depth_offset",
                                  &max_radius_depth_offset_);
        ReadOptionalValueFromYaml(fs, "stride_depth_offset", &stride_depth_offset_);
        ReadOptionalValueFromYaml(fs, "use_random_seed", &use_random_seed_);
        ReadOptionalValueFromYaml(fs, "image_size", &image_size_);
        ReadOptionalValueFromYaml(fs, "feature_config_path", &feature_config_path_);
        fs.release();

        // Process parameters
        if (model_path_ == "INFER_FROM_NAME")
            model_path_ = metafile_path_.parent_path() / (name_ + ".bin");
        else if (model_path_.is_relative())
            model_path_ = metafile_path_.parent_path() / model_path_;
        return true;
    }

    bool FeatureModel::GenerateModel()
    {
        // Generate camera poses
        std::vector<Transform3fA> camera2body_poses;
        GenerateGeodesicPoses(&camera2body_poses);

        std::shared_ptr<RendererGeometry> renderer_geometry_ptr;
        renderer_geometry_ptr = std::make_shared<RendererGeometry>("rg");
        renderer_geometry_ptr->SetUp();

        // Generate template views
        std::cout << "Start generating model " << name_ << std::endl;
        views_.resize(camera2body_poses.size());
        bool cancel = false;
        int count = 1;
        std::shared_ptr<FullTextureRenderer> renderer_ptr;
        if (!SetUpRenderer(renderer_geometry_ptr, &renderer_ptr))
            cancel = true;

        for (int i = 0; i < int(views_.size()); ++i)
        {
            if (cancel)
                continue;
            std::stringstream msg;
            msg << "Generate feature model " << name_ << ": view " << count++ << " of "
                << views_.size() << std::endl;
            std::cout << msg.str();

            // Render images
            renderer_ptr->set_camera2world_pose(camera2body_poses[i]);
            renderer_ptr->StartRendering();
            renderer_ptr->FetchTextureImage();
            renderer_ptr->FetchDepthImage();

            // Generate data
            views_[i].orientation =
                camera2body_poses[i].matrix().col(2).segment(0, 3);
            views_[i].data_points.resize(n_points_);
            if (!GeneratePointData(*renderer_ptr, camera2body_poses[i],
                                   &views_[i].data_points))
                cancel = true;
        }

        if (cancel)
            return false;
        std::cout << "Finish generating model " << name_ << std::endl;
        return true;
    }

    bool FeatureModel::LoadModel()
    {
        std::ifstream ifs{model_path_, std::ios::in | std::ios::binary};
        if (!ifs.is_open() || ifs.fail())
        {
            ifs.close();
            std::cout << "Could not open model file " << model_path_ << std::endl;
            return false;
        }

        if (!LoadModelParameters(kVersionID, kModelType, &ifs))
        {
            std::cout << "Model file " << model_path_
                      << " was generated using different model parameters" << std::endl;
            return false;
        }

        if (!LoadBodyData(&ifs))
        {
            std::cout << "Model file " << model_path_
                      << " was generated using different body parameters" << std::endl;
            return false;
        }

        // Load view data
        size_t n_views;
        ifs.read((char *)(&n_views), sizeof(n_views));
        views_.clear();
        views_.reserve(n_views);
        for (size_t i = 0; i < n_views; i++)
        {
            View tv;
            tv.data_points.resize(n_points_);
            ifs.read((char *)(tv.data_points.data()), n_points_ * sizeof(DataPoint));
            ifs.read((char *)(tv.orientation.data()), sizeof(tv.orientation));
            views_.push_back(std::move(tv));
        }
        ifs.close();
        return true;
    }

    bool FeatureModel::SaveModel()
    {
        std::ofstream ofs{model_path_, std::ios::out | std::ios::binary};
        SaveModelParameters(kVersionID, kModelType, &ofs);
        SaveBodyData(&ofs);

        // Save main data
        size_t n_views = views_.size();
        ofs.write((const char *)(&n_views), sizeof(n_views));
        for (const auto &v : views_)
        {
            ofs.write((const char *)(v.data_points.data()),
                      n_points_ * sizeof(DataPoint));
            ofs.write((const char *)(v.orientation.data()), sizeof(v.orientation));
        }
        ofs.flush();
        ofs.close();
        return true;
    }

    bool FeatureModel::GeneratePointData(const FullTextureRenderer &renderer,
                                         const Transform3fA &camera2body_pose,
                                         std::vector<DataPoint> *data_points)
    {
        // Extract kp and desc
        cv::Mat texture_image = renderer.texture_image();
        // Resize image to 400x400
        cv::resize(texture_image, texture_image, cv::Size(600, 600));
        cv::Mat depth_image;
        auto frame = WrapFrame(texture_image, depth_image);
        feature_manager_ptr_->detectFeature(frame);

        // Visualize
#define VISUALIZE_SPARSE_MODEL
#ifdef VISUALIZE_SPARSE_MODEL
        // Visualize keypoints
        for (auto &kp : frame->_keypts)
        {
            cv::circle(texture_image, kp.pt, 2, cv::Scalar(0, 0, 255), -1);
        }
        cv::imshow("texture", texture_image);
        cv::waitKey(0);
#endif

        // Calculate data for contour points
        int counter = 0;
        for (auto data_point{begin(*data_points)}; data_point != end(*data_points);)
        {
            if (counter >= frame->_num_feat)
                return true; // Early stop
            auto kp = frame->_keypts[counter];
            cv::Point2i center{kp.pt.x, kp.pt.y};
            Eigen::Vector3f center_f_camera{renderer.PointVector(center)};
            Eigen::Vector3f normal_f_camera{renderer.NormalVector(center)};
            data_point->center_f_body = camera2body_pose * center_f_camera;
            data_point->normal_f_body = camera2body_pose.rotation() * normal_f_camera;

            // Copy feature descriptor
            memcpy(data_point->descriptor.data(), (void *)(frame->_feat_des.ptr<float>(counter, 0)), sizeof(float) * frame->_feat_dim);

            data_point++;
            counter++;
        }
        return true;
    }

    std::shared_ptr<Frame> FeatureModel::WrapFrame(cv::Mat &color_image, cv::Mat &depth_image)
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

}
