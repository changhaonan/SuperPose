#pragma once
#include <icg/camera.h>
#include <icg/common.h>
#include <icg/viewer.h>

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
                      const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr);
        FeatureViewer(const std::string &name,
                      const std::filesystem::path &metafile_path,
                      const std::shared_ptr<ColorCamera> &color_camera_ptr,
                      const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr);
        bool SetUp() override;

        bool UpdateViewer(int save_index) override;

    private:
        // Helper method
        bool LoadMetaData();
        // Data
        std::shared_ptr<RendererGeometry> renderer_geometry_ptr_ = nullptr;
    };

}