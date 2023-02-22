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

    std::vector<cv::Point2f> PNPSolver::ProjectPoints(const std::vector<cv::Point3f> &object_points,
                                                      const Eigen::Matrix3f &rot_m, const Eigen::Vector3f &trans_m)
    {
        std::vector<cv::Point2f> projected_points;

        for (const auto &point_cv : object_points)
        {
            // From object to camera coordinate
            Eigen::Vector3f point_object;
            point_object << point_cv.x, point_cv.y, point_cv.z;
            Eigen::Vector3f point_camera = rot_m * point_object + trans_m;

            // Projection
            Eigen::Vector3f projected_point{point_camera(0) / point_camera(2), point_camera(1) / point_camera(2), 1};
            Eigen::Matrix3f K_m;
            cv::cv2eigen(K_, K_m);
            projected_point = K_m * projected_point;
            projected_points.push_back(cv::Point2f(projected_point(0), projected_point(1)));
        }
        return projected_points;
    }

    float PNPSolver::ProjectError(const std::vector<cv::Point3f> &object_points,
                                  const std::vector<cv::Point2f> &image_points,
                                  const Eigen::Matrix3f &rot_m, const Eigen::Vector3f &trans_m)
    {
        auto projected_points = ProjectPoints(object_points, rot_m, trans_m);
        float error = 0;
        for (int i = 0; i < projected_points.size(); i++)
        {
            error += cv::norm(projected_points[i] - image_points[i]);
        }
        return error / float(projected_points.size());
    }

    bool PNPSolver::PNPValid(const float &projection_error_before_pnp, const float &projection_error_after_pnp)
    {
        if ((projection_error_after_pnp < projection_error_before_pnp) && (projection_error_after_pnp < 5))
        {
            return true;
        }
        else
        {
            return false;
        }
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

        // Compute projection error
        float projection_error_before = ProjectError(object_points, image_points, rot_m, trans_m);

        if (cv::solvePnP(object_points, image_points, K_, cv::noArray(), rot_vec_cv, trans_vec_cv, use_extrinsic_guess, flags))
        {
            // Transform back
            cv::cv2eigen(trans_vec_cv, trans_m);
            cv::Mat rot_m_cv;
            cv::Rodrigues(rot_vec_cv, rot_m_cv);
            cv::cv2eigen(rot_m_cv, rot_m);

            float projection_error_after = ProjectError(object_points, image_points, rot_m, trans_m);
            std::cout << "EPnPSolver::SolvePNP: Projection error before: " << projection_error_before << std::endl;
            std::cout << "EPnPSolver::SolvePNP: Projection error after: " << projection_error_after << std::endl;
            // Do PNP rejection
            return PNPValid(projection_error_before, projection_error_after);
        }
        else
        {
            std::cout << "EPnPSolver::SolvePNP: Failed to solve PnP" << std::endl;
            return false;
        }
    }

} // namespace icg
