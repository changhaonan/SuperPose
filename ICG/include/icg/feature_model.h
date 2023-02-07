#pragma once
#include <filesystem/filesystem.h>
#include <icg/body.h>
#include <icg/common.h>
#include <icg/model.h>
#include <icg/normal_renderer.h>
#include <icg/renderer_geometry.h>
#include <omp.h>

namespace icg
{

    class FeatureModel : public Model
    {
    public:
        FeatureModel(const std::string &name,
                     const std::filesystem::path &metafile_path,
                     const std::shared_ptr<Body> &body_ptr);
        bool SetUp() override;

        struct DataPoint
        {
            Eigen::Vector3f center_f_body;
            Eigen::Vector3f normal_f_body;
            float foreground_distance = 0.0f;
            float background_distance = 0.0f;
            std::array<float, kMaxNDepthOffsets> depth_offsets{};
        };

        struct View
        {
            std::vector<DataPoint> data_points;
            Eigen::Vector3f orientation;
        };

    private:
        // Model data
        std::vector<View> views_;
    };

}