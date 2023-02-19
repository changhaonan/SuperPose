#include <icg/feature_model.h>
#include <opencv2/opencv.hpp>

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

        // Feature model has much less views than the full model
        // Use 12 views for the feature model
        set_n_divides(0);

        if (!metafile_path_.empty())
            if (!LoadMetaData())
                return false;

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

    bool FeatureModel::GetRelativeRotDeg(const Transform3fA &body2camera_pose,
                                         const View &view, float &relative_rot_deg) const
    {
        if (!set_up_)
        {
            std::cerr << "Set up feature model " << name_ << " first" << std::endl;
            return false;
        }

        // Project the x-axis of the camera to the body
        auto body_x_axis = body2camera_pose.rotation().inverse() * Eigen::Vector3f::UnitX();
        auto view_x_axis = view.rotation.inverse() * Eigen::Vector3f::UnitX();

        // Debug
        Eigen::Vector3f body_orientation{
            body2camera_pose.rotation().inverse() *
            body2camera_pose.translation().matrix().normalized()};
        std::cout << "body_orientation: " << body_orientation.transpose() << std::endl;
        std::cout << "view_orientation: " << view.orientation.transpose() << std::endl;

        // Check by printing
        std::cout << "body rotation" << std::endl;
        std::cout << body2camera_pose.rotation() << std::endl;
        std::cout << "view rotation" << std::endl;
        std::cout << view.rotation << std::endl;
        std::cout << "body_x_axis: " << body_x_axis.transpose() << std::endl;
        std::cout << "view_x_axis: " << view_x_axis.transpose() << std::endl;
        float dot = body_x_axis.dot(view_x_axis);
        relative_rot_deg = acos(dot) * 180.0f / M_PI;
        return true;
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
            renderer_ptr->FetchNormalImage();
            renderer_ptr->FetchDepthImage();

            // Generate data
            views_[i].orientation =
                camera2body_poses[i].matrix().col(2).segment(0, 3);
            views_[i].rotation = camera2body_poses[i].matrix().block(0, 0, 3, 3);
            if (!GenerateViewData(*renderer_ptr, camera2body_poses[i], views_[i]))
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
        size_t n_views, image_width, image_height;
        ifs.read((char *)(&n_views), sizeof(n_views));
        ifs.read((char *)(&image_width), sizeof(image_width));
        ifs.read((char *)(&image_height), sizeof(image_height));
        views_.clear();
        views_.reserve(n_views);
        for (size_t i = 0; i < n_views; i++)
        {
            View tv;
            // Load texture image
            tv.texture_image = cv::Mat(image_height, image_width, CV_8UC4);
            ifs.read((char *)(tv.texture_image.data), tv.texture_image.elemSize() * tv.texture_image.total());
            // Load depth image
            tv.depth_image = cv::Mat(image_height, image_width, CV_16FC1);
            ifs.read((char *)(tv.depth_image.data), tv.depth_image.elemSize() * tv.depth_image.total());
            // Load normal image
            tv.normal_image = cv::Mat(image_height, image_width, CV_8UC4);
            ifs.read((char *)(tv.normal_image.data), tv.normal_image.elemSize() * tv.normal_image.total());
            // Pose
            ifs.read((char *)(tv.orientation.data()), sizeof(tv.orientation));
            views_.push_back(std::move(tv));
            ifs.read((char *)(tv.rotation.data()), sizeof(tv.rotation));
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
        size_t image_width = views_[0].texture_image.cols;
        size_t image_height = views_[0].texture_image.rows;
        ofs.write((const char *)(&n_views), sizeof(n_views));
        ofs.write((const char *)(&image_width), sizeof(image_width));
        ofs.write((const char *)(&image_height), sizeof(image_height));
        for (const auto &v : views_)
        {
            ofs.write((const char *)v.texture_image.data, v.texture_image.total() * v.texture_image.elemSize());
            ofs.write((const char *)v.depth_image.data, v.depth_image.total() * v.depth_image.elemSize());
            ofs.write((const char *)v.normal_image.data, v.normal_image.total() * v.normal_image.elemSize());
            ofs.write((const char *)(v.orientation.data()), sizeof(v.orientation));
            ofs.write((const char *)(v.rotation.data()), sizeof(v.rotation));
        }
        ofs.flush();
        ofs.close();
        return true;
    }

    bool FeatureModel::GenerateViewData(const FullTextureRenderer &renderer, const Transform3fA &camera2body_pose,
                                        View &view)
    {
        // Extract kp and desc
        cv::Mat texture_image = renderer.texture_image();
        cv::Mat depth_image = renderer.depth_image();
        cv::Mat normal_image = renderer.normal_image();
        // Resize image to 400x400
        cv::resize(texture_image, texture_image, cv::Size(600, 600));
        cv::resize(depth_image, depth_image, cv::Size(600, 600));
        cv::resize(normal_image, normal_image, cv::Size(600, 600));

        // Save view data
        view.texture_image = texture_image.clone();
        view.depth_image = depth_image.clone();
        view.normal_image = normal_image.clone();
        return true;
    }

}
