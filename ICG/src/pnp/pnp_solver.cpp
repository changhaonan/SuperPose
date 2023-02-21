#include <icg/pnp/pnp_solver.h>

namespace icg
{
    PNPSolver::PNPSolver()
    {
    }

    PNPSolver::~PNPSolver()
    {
    }

    bool PNPSolver::SetUp(const std::filesystem::path &metafile_path)
    {
        // Open file storage from yaml
        cv::FileStorage fs;
        if (!OpenYamlFileStorage(metafile_path, &fs))
            return false;

        Intrinsics intrinsics;
        if (!ReadRequiredValueFromYaml(fs, "intrinsics", &intrinsics))
        {
            std::cerr << "Could not read all required body parameters from "
                      << metafile_path << std::endl;
            return false;
        }

        fs.release();

        // Setup instrinsic mat
        K_ = cv::Mat::eye(3, 3, CV_64F);
        K_.at<double>(0, 0) = intrinsics.fu;
        K_.at<double>(1, 1) = intrinsics.fv;
        K_.at<double>(0, 2) = intrinsics.ppu;
        K_.at<double>(1, 2) = intrinsics.ppv;

        return true;
    }

    EPnPSolver::EPnPSolver()
    {
    }

    EPnPSolver::~EPnPSolver()
    {
    }

    bool EPnPSolver::SolvePNP(const std::vector<cv::Point3f> &object_points,
                              const std::vector<cv::Point2f> &image_points,
                              Eigen::Matrix3f &rot_m, Eigen::Vector3f &trans_m, bool use_extrinsic_guess,
                              int flags)
    {
        // From rotation matrix to angle axis
        Eigen::AngleAxisf rot_vec(rot_m);
        Eigen::Vector3f rot_vec_3 = rot_vec.angle() * rot_vec.axis();

        cv::Mat rot_vec_cv;
        cv::eigen2cv(rot_vec_3, rot_vec_cv);
        cv::Mat trans_vec_cv;
        cv::eigen2cv(trans_m, trans_vec_cv);

        cv::solvePnP(object_points, image_points, K_, cv::noArray(), rot_vec_cv, trans_vec_cv, use_extrinsic_guess, flags);

        // Convert cv::Mat to Eigen::Matrix3f
        cv::cv2eigen(rot_vec_cv, rot_vec_3);
        cv::cv2eigen(trans_vec_cv, trans_m);

        // From angle axis to rotation matrix
        rot_m = Eigen::AngleAxisf(rot_vec_3.norm(), rot_vec_3.normalized()).toRotationMatrix();
        return true;
    }

} // namespace icg
