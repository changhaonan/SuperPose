#pragma once
#include <filesystem/filesystem.h>
#include <icg/body.h>
#include <icg/camera.h>
#include <icg/common.h>
#include <icg/modality.h>
#include <icg/feature_manager.h>
#include <icg/feature_model.h>

namespace icg
{

    class FeatureModality : public Modality
    {
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

        struct DataPoint
        {
        };

        // Main methods
        bool StartModality(int iteration, int corr_iteration) override;
        bool CalculateCorrespondences(int iteration, int corr_iteration) override;
        bool VisualizeCorrespondences(int save_idx) override;
        bool CalculateGradientAndHessian(int iteration, int corr_iteration,
                                         int opt_iteration) override;
        bool VisualizeOptimization(int save_idx) override;
        bool CalculateResults(int iteration) override;
        bool VisualizeResults(int save_idx) override;

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

        // Visualization method
        bool VisualizeCorrespondences(const std::string &title, int save_idx);

    private:
        // Helper method for setup
        bool LoadMetaData();

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

        // Parameters to turn on individual visualizations
        bool visualize_correspondences_correspondence_ = false;
        bool visualize_points_correspondence_ = false;
        bool visualize_points_depth_rendering_correspondence_ = false;
        bool visualize_points_optimization_ = false;
        bool visualize_points_result_ = false;

        // Feature manager related
        int port_;
        std::string feature_config_file_;
        std::shared_ptr<NetworkFeature> feature_manager_ptr_ = nullptr;
        std::shared_ptr<Frame> current_frame_ptr_ = nullptr;

        // Internal states
        bool depth_enabled_ = false;
    };

}