#pragma once
#include <filesystem/filesystem.h>
#include <icg/body.h>
#include <icg/common.h>
#include <icg/model.h>
// #include <icg/normal_renderer.h>
#include <icg/renderer_geometry.h>
#include <icg/texture_renderer.h>
#include <Eigen/Eigen>
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
        // static constexpr char kFeatureType = 'r';  // 'r' for R2D2, 's' for superpoint, 'o' for orb
        // static constexpr int kDescriptorDim = 128; // Use R2D2

    public:
        FeatureModel(const std::string &name,
                     const std::filesystem::path &metafile_path,
                     const std::shared_ptr<Body> &body_ptr);
        bool SetUp() override;

        /**
         * \brief Struct that contains all data that is generated from the rendered
         * geometry of a body for a specific viewpoint and that is used by the \ref
         * FeatureModel.
         * @param texture_image color image of the rendered geometry.
         * @param depth_image depth image of the rendered geometry
         * @param normal_image normal image of the rendered geometry.
         * @param orientation vector that points from the camera center to the body
         * center.
         */
        struct View
        {
            cv::Mat texture_image;
            cv::Mat depth_image;
            cv::Mat normal_image;
            Eigen::Vector3f orientation;
            Eigen::Matrix3f rotation; // Rotation from camera to body
        };

        // Main methods
        bool GetClosestView(const Transform3fA &body2camera_pose,
                            const View **closest_view) const;
        // Helper methods to get better matching
        bool GetRelativeRotDeg(const Transform3fA &body2camera_pose,
                               const View &view, float &relative_rot_deg) const;

    private:
        // Helper methods for model set up
        bool LoadMetaData();
        bool GenerateModel();
        bool LoadModel();
        bool SaveModel();

        // Helper methods for view generation
        bool GenerateViewData(const FullTextureRenderer &renderer, const Transform3fA &camera2body_pose,
                              View &view);
        // Model data
        std::vector<View> views_;
    };

}