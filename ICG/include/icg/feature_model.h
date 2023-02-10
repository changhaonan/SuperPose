#pragma once
#include <filesystem/filesystem.h>
#include <icg/body.h>
#include <icg/common.h>
#include <icg/model.h>
// #include <icg/normal_renderer.h>
#include <icg/renderer_geometry.h>
#include <icg/texture_renderer.h>
#include <icg/feature_manager.h>
#include <omp.h>

namespace icg
{
    /**
     * \brief \ref Model that holds a *Sparse Viewpoint Model* that is generated
     * from a \ref Body and that is used by the \ref FeatureModality.
     *
     * \details For each viewpoint, the object stores a \ref View object that
     * includes data for all sampled feature points. Given the `body2camera_pose`
     * the closest view can be accessed using `GetClosestView()`.
     */
    class FeatureModel : public Model
    {
    private:
        // Model definition
        static constexpr char kModelType = 'f';
        static constexpr int kVersionID = 6;

        // Some constants
        static constexpr char kFeatureType = 'r';  // 'r' for R2D2, 's' for superpoint, 'o' for orb
        static constexpr int kDescriptorDim = 128; // Use R2D2

    public:
        FeatureModel(const std::string &name,
                     const std::filesystem::path &metafile_path,
                     const std::shared_ptr<Body> &body_ptr);
        bool SetUp() override;
        /**
         * \brief Struct that contains all data related to a contour point and that is
         * used by the \ref FeatureModel.
         * @param center_f_body 3D feature point.
         * @param normal_f_body 3D surface normal vector at feature point location.
         * @param depth_offsets differences between the depth value of the
         * `center_f_body` coordinate and the minimum depth value within a quadratic
         * kernel for which radius values are increasing by the parameter
         * `stride_depth_offset`.
         */
        struct DataPoint
        {
            Eigen::Vector3f center_f_body;
            Eigen::Vector3f normal_f_body;
            std::array<float, kDescriptorDim> descriptor{};
        };

        /**
         * \brief Struct that contains all data that is generated from the rendered
         * geometry of a body for a specific viewpoint and that is used by the \ref
         * FeatureModel.
         * @param data_points vector of all contour point information.
         * @param orientation vector that points from the camera center to the body
         * center.
         */
        struct View
        {
            std::vector<DataPoint> data_points;
            Eigen::Vector3f orientation;
        };

        // Main methods
        bool GetClosestView(const Transform3fA &body2camera_pose,
                            const View **closest_view) const;

        // Share feature manager
        std::shared_ptr<NetworkFeature> feature_manager_ptr() const;

        // Shared Processing method
        static std::shared_ptr<Frame> WrapFrame(cv::Mat &color_image, cv::Mat &depth_image);

    private:
        // Helper methods for model set up
        bool LoadMetaData();
        bool GenerateModel();
        bool LoadModel();
        bool SaveModel();

        // Helper methods for view generation
        bool GeneratePointData(const FullTextureRenderer &renderer,
                               const Transform3fA &camera2body_pose,
                               std::vector<DataPoint> *data_points);
        // Model data
        std::vector<View> views_;
        // Feature manager
        std::shared_ptr<NetworkFeature> feature_manager_ptr_;
        std::string feature_config_path_;
    };

}