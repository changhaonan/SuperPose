// This app is used to test the function & structure of ICG

#include <filesystem/filesystem.h>
#include <icg/generator.h>
#include <icg/tracker.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Not enough arguments: Provide configfile_path";
        return 0;
    }
//     const std::filesystem::path configfile_path{argv[1]};
//     std::shared_ptr<icg::Tracker> tracker_ptr;
//     if (!GenerateConfiguredTracker(configfile_path, &tracker_ptr)) return 0;

//     if (!tracker_ptr->SetUp()) return 0;
//     // Test the viewer

//     bool tracking_started = false;
//     bool quit_tracker_process = false;
//     bool execute_detection = true;
//     bool start_tracking = true;
//     std::vector<icg::Transform3fA> poses;
//     tracker_ptr->model_ptrs()[0]->GenerateGeodesicPoses(&poses);
//     std::cout << "Camera Poses:" << std::endl;
//     std::cout << poses[0].matrix() << std::endl;
//     for (int iteration = 0;; ++iteration) {
//         auto begin{std::chrono::high_resolution_clock::now()};
//         if (!tracker_ptr->ExecuteDetectionCycle(iteration)) return false;  // Run detection
//         if (!tracker_ptr->UpdateCameras(execute_detection)) return false;
//         if (!tracker_ptr->UpdateViewers(iteration)) return false;
//   }
}