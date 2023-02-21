#pragma once
#include <icg/camera.h>
#include <icg/common.h>
#include <icg/viewer.h>
#include <icg/renderer_geometry.h>
#include <icg/texture_renderer.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

namespace icg
{

    /**
     * \brief \ref Viewer that displays color images from a \ref ColorCamera.
     */
    class FeatureViewer : public ColorViewer
    {
    public:
        FeatureViewer(const std::string &name,
                      const std::shared_ptr<ColorCamera> &color_camera_ptr,
                      const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
                      float opacity = 0.5f);
        FeatureViewer(const std::string &name,
                      const std::filesystem::path &metafile_path,
                      const std::shared_ptr<ColorCamera> &color_camera_ptr,
                      const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
                      float opacity = 0.5f);
        bool SetUp() override;

        // Setters
        void set_renderer_geometry_ptr(
            const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr);
        void set_opacity(float opacity);

        // Main methods
        bool UpdateViewer(int save_index) override;

        // Getters
        std::shared_ptr<RendererGeometry> renderer_geometry_ptr() const override;
        float opacity() const;

    private:
        // Helper method
        bool LoadMetaData();
        cv::Mat CalculateAlphaBlend(const cv::Mat &camera_image,
                                    const cv::Mat &renderer_image, float opacity);
        float opacity_;
        std::shared_ptr<RendererGeometry> renderer_geometry_ptr_;
        FullTextureRenderer renderer_;
    };

}