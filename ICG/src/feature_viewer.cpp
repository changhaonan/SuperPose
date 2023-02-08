#include <icg/feature_viewer.h>

namespace icg
{
    FeatureViewer::FeatureViewer(
        const std::string &name,
        const std::shared_ptr<ColorCamera> &color_camera_ptr)
        : ColorViewer{name, color_camera_ptr} {}

    FeatureViewer::FeatureViewer(
        const std::string &name, const std::filesystem::path &metafile_path,
        const std::shared_ptr<ColorCamera> &color_camera_ptr)
        : ColorViewer{name, metafile_path, color_camera_ptr} {}

    bool FeatureViewer::SetUp()
    {
        set_up_ = false;
        if (!metafile_path_.empty())
            if (!LoadMetaData())
                return false;

        // Check if all required objects are set up
        if (!color_camera_ptr_->set_up())
        {
            std::cerr << "Color camera " << color_camera_ptr_->name()
                      << " was not set up" << std::endl;
            return false;
        }
        set_up_ = true;
        return true;
    }

    bool FeatureViewer::UpdateViewer(int save_index)
    {
        if (!set_up_)
        {
            std::cerr << "Set up image color viewer " << name_ << " first" << std::endl;
            return false;
        }
        DisplayAndSaveImage(save_index, color_camera_ptr_->image());
        return true;
    }

    bool FeatureViewer::LoadMetaData()
    { // Open file storage from yaml
        cv::FileStorage fs;
        if (!OpenYamlFileStorage(metafile_path_, &fs))
            return false;

        // Read parameters from yaml
        ReadOptionalValueFromYaml(fs, "display_images", &display_images_);
        ReadOptionalValueFromYaml(fs, "save_images", &save_images_);
        ReadOptionalValueFromYaml(fs, "save_directory", &save_directory_);
        ReadOptionalValueFromYaml(fs, "save_image_type", &save_image_type_);
        fs.release();

        // Process parameters
        if (save_directory_.is_relative())
            save_directory_ = metafile_path_.parent_path() / save_directory_;
        return true;
    }
}