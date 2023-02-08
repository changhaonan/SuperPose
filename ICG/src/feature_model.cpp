#include <icg/feature_model.h>

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
        set_up_ = true;
        return true;
    }

}
