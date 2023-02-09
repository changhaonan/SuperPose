// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Manuel Stoiber, German Aerospace Center (DLR)

#ifndef ICG_INCLUDE_ICG_RENDERER_GEOMETRY_H_
#define ICG_INCLUDE_ICG_RENDERER_GEOMETRY_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <icg/body.h>
#include <icg/common.h>

#include <filesystem/filesystem.h>
#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace icg {

/**
 * \brief Loads geometric information from referenced \ref Body objects in
 * *Vertex Array Objects (VAOs)* and *Vertex Buffers Objects (VBOs)* and
 * provides this information together with a *GLFW* context required for
 * rendering.
 *
 * \details Rendering data with *VAOs* and *VBOs* for all assigned
 * bodies is stored in `render_data_bodies`. The *GLFW* context can be handled
 * using `MakeContextCurrent()` and `DetachContext()`. Both the destructor and
 * the `SetUp()` method have to be called from the main thread to comply with
 * *GLFW* thread-safety requirements. Calling the `SetUp()` method initializes
 * *GLFW*, creates a *GLFW* context, and creates *VAOs* and *VBOs*. \ref Body
 * objects can be added and deleted without requiring a new call of `SetUp()`.
 * Setters and all main methods are thread-safe.
 *
 * @param body_ptrs referenced \ref Body objects that are considered in the
 * renderer geometry.
 */
class RendererGeometry {
 private:
  // Count the number of instances to manage the GLFW library
  static int n_instances_;

 public:
  // Data Structs
  struct RenderDataBody {
    Body *body_ptr = nullptr;
    GLuint vao = 0;
    GLuint vbo = 0;
    unsigned n_vertices = 0;
    GLuint texture = 0;
  };

  // Constructor, destructor, and setup method
  RendererGeometry(const std::string &name);
  ~RendererGeometry();  // deletes glfw context
  bool SetUp();         // creates glfw context

  // Configure bodies
  bool AddBody(const std::shared_ptr<Body> &body_ptr);
  bool DeleteBody(const std::string &name);
  void ClearBodies();

  // Handling of GLFW context
  bool MakeContextCurrent();
  bool DetachContext();

  // Getters
  const std::string &name() const;
  const std::vector<std::shared_ptr<Body>> &body_ptrs() const;
  const std::vector<RenderDataBody> &render_data_bodies() const;
  bool set_up() const;
  bool enable_texture() const;

 private:
  // Helper methods
  static void AssembleVertexData(const Body &body,
                                 std::vector<float> *vertex_data);
  static void AssembleVertexDataWithTexture(const Body &body,
                                 std::vector<float> *vertex_data);
  static void CreateGLVertexObjects(const std::vector<float> &vertices,
                                    RenderDataBody *render_data_body);
  static void CreateGLVertexObjectsWithTexture(const std::vector<float> &vertices,
                                    RenderDataBody *render_data_body);
  static void DeleteGLVertexObjects(RenderDataBody *render_data_body);
  static void CreateGLTextureObjects(const Body &body,
                                   RenderDataBody *render_data_body);
  // Variables
  std::string name_{};
  std::vector<std::shared_ptr<Body>> body_ptrs_;
  std::vector<RenderDataBody> render_data_bodies_;
  GLFWwindow *window_ = nullptr;  // Only used to hold a glfw context
  std::mutex mutex_;
  bool initial_set_up_ = false;
  bool set_up_ = false;
  bool enable_texture_ = true;
};

}  // namespace icg

#endif  // ICG_INCLUDE_ICG_RENDERER_GEOMETRY_H_
