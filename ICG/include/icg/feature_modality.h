#pragma once
#include <filesystem/filesystem.h>
#include <icg/body.h>
#include <icg/camera.h>
#include <icg/common.h>
#include <icg/modality.h>
#include <icg/feature_model.h>
#include <icg/pnp/pnp_solver.h>

namespace pfh
{
    class MatcherClient;
}

namespace icg
{

    class FeatureModality : public Modality
    {
    private:
        // Data for correspondence point calculated during `CalculateCorrespondences`
        struct DataPoint
        {
            // Eigen::Vector3f center_f_body{};
            // Eigen::Vector3f center_f_camera{};
            // Eigen::Vector3f normal_f_body{};
            // float center_u = 0.0f;
            // float center_v = 0.0f;
            // float depth = 0.0f;
            // float measured_depth_offset = 0.0f;
            // float modeled_depth_offset = 0.0f;
            // Eigen::Vector3f correspondence_center_f_camera{};
            Eigen::Vector3f body_point;
            Eigen::Vector3f body_normal;
            cv::Point2i body_uv;
            cv::Point2i camera_uv;
        };

    public:
        FeatureModality(const std::string &name, const std::shared_ptr<Body> &body_ptr,
                        const std::shared_ptr<ColorCamera> &color_camera_ptr,
                        const std::shared_ptr<DepthCamera> &depth_camera_ptr,
                        const std::shared_ptr<FeatureModel> &feature_model_ptr);
        FeatureModality(
            const std::string &name, const std::filesystem::path &metafile_path,
            const std::shared_ptr<Body> &body_ptr,
            const std::shared_ptr<ColorCamera> &color_camera_ptr,
            const std::shared_ptr<DepthCamera> &depth_camera_ptr,
            const std::shared_ptr<FeatureModel> &feature_model_ptr);
        bool SetUp() override;

        // using DataPoint = FeatureModel::DataPoint;

        // Main methods
        bool StartModality(int iteration, int corr_iteration) override;
        bool CalculateCorrespondences(int iteration, int corr_iteration) override;
        bool VisualizeCorrespondences(int save_idx) override;
        bool CalculateGradientAndHessian(int iteration, int corr_iteration,
                                         int opt_iteration) override;
        bool VisualizeOptimization(int save_idx) override;
        bool CalculateResults(int iteration) override;
        bool VisualizeResults(int save_idx) override;

        // Related with how to integrate with feature matching
        bool RunPNP();

        // Getters data
        const std::shared_ptr<ColorCamera> &color_camera_ptr() const;
        const std::shared_ptr<FeatureModel> &feature_model_ptr() const;
        std::shared_ptr<Model> model_ptr() const override;
        std::vector<std::shared_ptr<Camera>> camera_ptrs() const override;
        std::vector<std::shared_ptr<Renderer>> start_modality_renderer_ptrs()
            const override;
        std::vector<std::shared_ptr<Renderer>> correspondence_renderer_ptrs()
            const override;
        std::vector<std::shared_ptr<Renderer>> results_renderer_ptrs() const override;
        int n_points() const;

        // Setters
        void set_n_points(int n_points);

        // Visualization method
        bool VisualizeCorrespondences(const std::string &title, int save_idx);

    private:
        // Helper method for setup
        bool LoadMetaData();

        // Helper methods for precalculation of referenced data and changing data
        void PrecalculateCameraVariables();
        bool PrecalculateModelVariables();
        void PrecalculateRendererVariables();
        void PrecalculatePoseVariables();
        void PrecalculateIterationDependentVariables(int corr_iteration);

        // Helper methods for CalculateCorrespondences
        bool MatchFeatures(const cv::Mat &view_descriptor, const cv::Mat &frame_descriptor, std::vector<cv::DMatch> &matches, float ratio_thresh = 0.8f);
        void ComputeCurrentROI();
        void CalculateBasicPointData(DataPoint *data_point, const FeatureModel::View &view,
                                     const cv::KeyPoint &body_kps,
                                     const cv::KeyPoint &camera_kps) const;

        // Helper method for visualization
        void VisualizePointsFeatureImage(const std::string &title, int save_idx) const;
        void ShowAndSaveImage(const std::string &title, int save_idx, const cv::Mat &image) const;
        // Other helper methods
        bool IsSetup() const;

        // Internal data objects
        std::vector<DataPoint> data_points_;

        // Pointers to referenced objects
        std::shared_ptr<ColorCamera> color_camera_ptr_ = nullptr;
        std::shared_ptr<DepthCamera> depth_camera_ptr_ = nullptr;
        std::shared_ptr<FeatureModel> feature_model_ptr_ = nullptr;
        std::shared_ptr<FocusedDepthRenderer> depth_renderer_ptr_ = nullptr;

        // Parameters for general distribution
        int n_points_ = 200;

        // Parameters to turn on individual visualizations
        bool visualize_correspondences_correspondence_ = false;
        bool visualize_points_correspondence_ = false;
        bool visualize_points_depth_rendering_correspondence_ = false;
        bool visualize_points_optimization_ = false;
        bool visualize_points_result_ = false;

        // Feature matching related
        int port_;
        Eigen::Vector4f current_roi_ = Eigen::Vector4f::Zero();
        int roi_margin_ = 5;
        std::shared_ptr<pfh::MatcherClient> matcher_client_ptr_ = nullptr;

        // Internal states
        bool depth_enabled_ = false;

        // Precalculated variables for camera (referenced data)
        float fu_{};
        float fv_{};
        float ppu_{};
        float ppv_{};
        int image_width_minus_1_{};
        int image_height_minus_1_{};
        float depth_scale_{};

        // Precalculated variables for poses (continuously changing)
        Transform3fA body2camera_pose_{};
        Transform3fA camera2body_pose_{};
        Eigen::Matrix3f body2camera_rotation_{};

        // PNP solver
        std::shared_ptr<PNPSolver> pnp_solver_ptr_ = nullptr;
    };

}