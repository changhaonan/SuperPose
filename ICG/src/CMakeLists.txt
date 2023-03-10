cmake_minimum_required(VERSION 3.10)

# Define sources
# =============================================================================
set(SOURCES
        common.cpp body.cpp renderer_geometry.cpp renderer.cpp
        normal_renderer.cpp basic_depth_renderer.cpp model.cpp region_model.cpp
        depth_model.cpp camera.cpp loader_camera.cpp viewer.cpp
        image_viewer.cpp normal_viewer.cpp modality.cpp region_modality.cpp
        depth_modality.cpp optimizer.cpp detector.cpp static_detector.cpp network_detector.cpp
        manual_detector.cpp refiner.cpp publisher.cpp tracker.cpp
        frame.cpp feature_manager.cpp
        feature_modality.cpp feature_model.cpp feature_viewer.cpp texture_renderer.cpp
        generator.cpp)

if (USE_AZURE_KINECT)
    list(APPEND SOURCES azure_kinect_camera.cpp)
endif ()
if (USE_REALSENSE)
    list(APPEND SOURCES realsense_camera.cpp)
endif ()

set(PNP_SOURCES pnp/pnp_solver.cpp)

# Define libraries
# =============================================================================
set(LIBRARIES Eigen3::Eigen OpenGL::GL GLEW::GLEW glfw ${OpenCV_LIBRARIES} zmq yaml-cpp)
if (USE_AZURE_KINECT)
    list(APPEND LIBRARIES k4a::k4a)
endif ()
if (USE_REALSENSE)
    list(APPEND LIBRARIES ${realsense2_LIBRARY})
endif ()
if(UNIX)
    list(APPEND LIBRARIES pthread stdc++fs)
endif()

# Define target
# =============================================================================
add_library(pnp STATIC ${PNP_SOURCES})
target_link_libraries(pnp PRIVATE Eigen3::Eigen ${OpenCV_LIBRARIES})

add_library(icg STATIC ${SOURCES})
target_link_libraries(icg PUBLIC ${LIBRARIES} pnp)
