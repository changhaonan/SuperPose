// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Manuel Stoiber, German Aerospace Center (DLR)

#ifndef ICG_INCLUDE_ICG_MODEL_H_
#define ICG_INCLUDE_ICG_MODEL_H_

#include <filesystem/filesystem.h>
#include <icg/body.h>
#include <icg/common.h>
#include <icg/normal_renderer.h>
#include <icg/texture_renderer.h>
#include <icg/renderer_geometry.h>
#include <omp.h>

#include <filesystem/filesystem.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <atomic>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

namespace icg
{

    /**
     * \brief Abstract class that precomputes and stores geometric information from
     * \ref Body objects that is required by \ref Modality objects during tracking.
     *
     * \details Typically, the class creates a *Sparse Viewpoint Model* of the \ref
     * Body, which stores relevant information for all possible viewpoints from
     * virtual cameras around the object.
     *
     * @param body_ptr referenced \ref Body object for which the model is generated.
     * @param model_path path that specifies the location of the model file. If the
     * model file does not exist or different parameters were used, it is generated
     * again. Using INFER_FROM_NAME in the metafile sets the path to <name>.bin.
     * @param sphere_radius distance from the object center to the center of a
     * virtual camera.
     * @param n_divides number of times an icosahedron is divided to generate
     * viewpoints on a sphere. It thereby controls the number of template views.
     * @param n_points number of points that are sampled for each view.
     * @param max_radius_depth_offset maximum radius in meter for the calculation of
     * `depth_offsets` that are used for occlusion handling.
     * @param stride_depth_offset distance between points in meter that are used to
     * calculate `depth_offsets`.
     * @param use_random_seed if a random seed is used to sample points.
     * @param image_size size of images that are rendered for each view.
     */
    class Model
    {
    protected:
        // Some constants
        static constexpr int kImageSizeSafetyBoundary = 20;
        static constexpr int kMaxNDepthOffsets = 30;

        // Struct with operator that compares two Vector3f and checks if v1 < v2
        struct CompareSmallerVector3f
        {
            bool operator()(const Eigen::Vector3f &v1,
                            const Eigen::Vector3f &v2) const
            {
                return v1[0] < v2[0] || (v1[0] == v2[0] && v1[1] < v2[1]) ||
                       (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] < v2[2]);
            }
        };

    public:
        // Setup methods
        virtual bool SetUp() = 0;

        // Setters
        void set_name(const std::string &name);
        void set_metafile_path(const std::filesystem::path &metafile_path);
        void set_body_ptr(const std::shared_ptr<Body> &body_ptr);
        void set_model_path(const std::filesystem::path &model_path);
        void set_sphere_radius(float sphere_radius);
        /// Number of subdivisions of an icosahedron to obtain camera positions.
        void set_n_divides(int n_divides);
        void set_n_points(int n_points);
        void set_max_radius_depth_offset(float max_radius_depth_offset);
        void set_stride_depth_offset(float stride_depth_offset);
        void set_use_random_seed(bool use_random_seed);
        void set_image_size(int image_size);

        // Getters
        const std::string &name() const;
        const std::filesystem::path &metafile_path() const;
        const std::shared_ptr<Body> &body_ptr() const;
        const std::filesystem::path &model_path() const;
        float sphere_radius() const;
        int n_divides() const;
        int n_points() const;
        float max_radius_depth_offset() const;
        float stride_depth_offset() const;
        bool use_random_seed() const;
        int image_size() const;
        bool set_up() const;

    protected:
        // Constructor
        Model(const std::string &name, const std::shared_ptr<Body> &body_ptr,
              const std::filesystem::path &model_path, float sphere_radius,
              int n_divides, int n_points, float max_radius_depth_offset,
              float stride_depth_offset, bool use_random_seed, int image_size);
        Model(const std::string &name, const std::filesystem::path &metafile_path,
              const std::shared_ptr<Body> &body_ptr);

        // Helper methods for view generation
        bool SetUpRenderer(
            const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
            std::shared_ptr<FullNormalRenderer> *renderer) const;
        bool SetUpRenderer(
            const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
            std::shared_ptr<FullTextureRenderer> *renderer) const;

        // Helper methods to save and load data
        bool LoadModelParameters(int version_id, char model_type, std::ifstream *ifs);
        bool LoadBodyData(std::ifstream *ifs);
        void SaveModelParameters(int version_id, char model_type,
                                 std::ofstream *ofs) const;
        void SaveBodyData(std::ofstream *ofs) const;

        // Helper methods for view generation
        bool DepthOffsetVariablesValid() const;
        void CalculateDepthOffsets(
            const FullNormalRenderer &renderer, const cv::Point2i &center,
            float pixel_to_meter,
            std::array<float, kMaxNDepthOffsets> *depth_offsets) const;

        // Helper methods to generate geodesic poses
        void GenerateGeodesicPoses(
            std::vector<Transform3fA> *camera2body_poses) const;
        void GenerateGeodesicPoints(
            std::set<Eigen::Vector3f, CompareSmallerVector3f> *geodesic_points) const;
        static void SubdivideTriangle(
            const Eigen::Vector3f &v1, const Eigen::Vector3f &v2,
            const Eigen::Vector3f &v3, int n_divides,
            std::set<Eigen::Vector3f, CompareSmallerVector3f> *geodesic_points);

        // Variables and data
        std::string name_{};
        std::filesystem::path metafile_path_{};
        std::shared_ptr<Body> body_ptr_ = nullptr;
        std::filesystem::path model_path_{};
        float sphere_radius_ = 0.8f;
        int n_divides_ = 4;
        int n_points_ = 200;
        float max_radius_depth_offset_ = 0.05f;
        float stride_depth_offset_ = 0.002f;
        bool use_random_seed_ = false;
        int image_size_ = 2000;
        bool set_up_ = false;
    };

} // namespace icg

#endif // ICG_INCLUDE_ICG_MODEL_H_
