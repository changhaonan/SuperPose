#include <string>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <icg/common.h>
#include <filesystem/filesystem.h>

namespace icg
{

    class PNPSolver
    {
    public:
        enum PnpSolverType
        {
            PNP_SOLVER_ITERATIVE = 0,
            PNP_SOLVER_EPNP = 1,
            PNP_SOLVER_P3P = 2,
            PNP_SOLVER_DLS = 3,
            PNP_SOLVER_UPNP = 4,
            PNP_SOLVER_AP3P = 5,
            PNP_SOLVER_MAX_COUNT = 6
        };

        PNPSolver();
        ~PNPSolver();
        // Setup with camera metafile
        bool SetUp(const std::filesystem::path &metafile_path);

        /**
         * @brief SolvePNP
         * @param object_points
         * @param image_points
         * @param K
         * @param dist_coeffs
         * @param rvec
         * @param tvec
         * @param use_extrinsic_guess
         * @param iterations_count
         * @param reprojection_error
         * @param confidence
         * @param inliers
         * @param flags
         * @return
         * @see cv::solvePnP
         */
        virtual bool SolvePNP(const std::vector<cv::Point3f> &object_points,
                              const std::vector<cv::Point2f> &image_points,
                              Eigen::Matrix3f &rot_m, Eigen::Vector3f &trans_m, bool use_extrinsic_guess,
                              int flags = cv::SOLVEPNP_ITERATIVE) = 0;
                              
        // Projection
        std::vector<cv::Point2f> ProjectPoints(const std::vector<cv::Point3f> &object_points,
                                               const Eigen::Matrix3f &rot_m, const Eigen::Vector3f &trans_m);

        float ProjectError(const std::vector<cv::Point3f> &object_points,
                           const std::vector<cv::Point2f> &image_points,
                           const Eigen::Matrix3f &rot_m, const Eigen::Vector3f &trans_m);
        bool PNPValid(const float & projection_error_before_pnp, const float & projection_error_after_pnp);

        cv::Mat K_;
    };

    class EPnPSolver : public PNPSolver
    {
    public:
        EPnPSolver();
        ~EPnPSolver();

        bool SolvePNP(const std::vector<cv::Point3f> &object_points,
                      const std::vector<cv::Point2f> &image_points,
                      Eigen::Matrix3f &rot_m, Eigen::Vector3f &trans_m, bool use_extrinsic_guess,
                      int flags = cv::SOLVEPNP_ITERATIVE) override;
    };
}