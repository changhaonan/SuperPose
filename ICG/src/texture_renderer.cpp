// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Manuel Stoiber, German Aerospace Center (DLR)

#include <icg/texture_renderer.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace icg
{

    std::string TextureRendererCore::vertex_shader_code_ =
        "#version 330 core\n"
        "layout(location = 0) in vec3 aPos;\n"
        "layout(location = 1) in vec3 aNormal;\n"
        "layout(location = 2) in vec2 aTexCoord;\n"
        "out VertexOut {\n"
        "   vec3 normal;\n"
        "   vec2 texcoord;\n"
        "   flat int vertex_id;\n"
        "} vertex_shader_out;\n"
        "uniform mat4 Trans;\n"
        "uniform mat3 Rot;\n"
        "void main()\n"
        "{\n"
        "  gl_Position = Trans * vec4(aPos, 1.0);\n"
        "  vertex_shader_out.texcoord = aTexCoord;\n"
        "  vertex_shader_out.normal = Rot * aNormal;\n"
        "}";

    std::string TextureRendererCore::fragment_shader_code_ =
        "#version 330 core\n"
        "in VertexOut {\n"
        "   vec3 normal;\n"
        "   vec2 texcoord;\n"
        "   flat int vertex_id;\n"
        "} vertex_shader_in;\n"
        "layout(location = 0) out vec4 FragColor0;\n"
        "layout(location = 1) out vec4 FragColor1;\n"
        "uniform sampler2D Texture;\n"
        "void main()\n"
        "{\n"
        "	gl_FragData[0] = texture(Texture, vertex_shader_in.texcoord);\n"
        "   gl_FragData[1] = vec4(0.5 - 0.5 * vertex_shader_in.normal, 1.0).zyxw;\n"
        "}";

    TextureRendererCore::~TextureRendererCore()
    {
        if (initial_set_up_)
            DeleteBufferObjects();
    }

    bool TextureRendererCore::SetUp(
        const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
        int image_width, int image_height)
    {
        renderer_geometry_ptr_ = renderer_geometry_ptr;
        image_width_ = image_width;
        image_height_ = image_height;
        image_rendered_ = false;

        // Create shader program
        if (!initial_set_up_ &&
            !CreateShaderProgram(renderer_geometry_ptr_.get(),
                                 vertex_shader_code_.c_str(),
                                 fragment_shader_code_.c_str(), &shader_program_))
            return false;

        // Create buffer objects
        if (initial_set_up_)
            DeleteBufferObjects();
        CreateBufferObjects();
        initial_set_up_ = true;
        return true;
    }

    bool TextureRendererCore::StartRendering(
        const Eigen::Matrix4f &projection_matrix,
        const Transform3fA &world2camera_pose)
    {
        if (!initial_set_up_)
            return false;
        renderer_geometry_ptr_->MakeContextCurrent();
        glViewport(0, 0, image_width_, image_height_);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glFrontFace(GL_CCW);
        glCullFace(GL_FRONT);

        glUseProgram(shader_program_);
        for (const auto &render_data_body :
             renderer_geometry_ptr_->render_data_bodies())
        {
            Transform3fA trans_without_projection{
                world2camera_pose * render_data_body.body_ptr->geometry2world_pose()};
            Eigen::Matrix4f trans{projection_matrix *
                                  trans_without_projection.matrix()};
            Eigen::Matrix3f rot{trans_without_projection.rotation().matrix()};

            unsigned loc;
            loc = glGetUniformLocation(shader_program_, "Trans");
            glUniformMatrix4fv(loc, 1, GL_FALSE, trans.data());
            loc = glGetUniformLocation(shader_program_, "Rot");
            glUniformMatrix3fv(loc, 1, GL_FALSE, rot.data());

            if (render_data_body.body_ptr->geometry_enable_culling())
                glEnable(GL_CULL_FACE);
            else
                glDisable(GL_CULL_FACE);

            glBindTexture(GL_TEXTURE_2D, render_data_body.texture);
            glBindVertexArray(render_data_body.vao);
            glDrawArrays(GL_TRIANGLES, 0, render_data_body.n_vertices);
            glBindVertexArray(0);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        renderer_geometry_ptr_->DetachContext();

        image_rendered_ = true;
        texture_image_fetched_ = false;
        normal_image_fetched_ = false;
        depth_image_fetched_ = false;
        return true;
    }

    bool TextureRendererCore::FetchTextureImage(cv::Mat *texture_image)
    {
        if (!initial_set_up_ || !image_rendered_)
            return false;
        if (texture_image_fetched_)
            return true;
        renderer_geometry_ptr_->MakeContextCurrent();
        glPixelStorei(GL_PACK_ALIGNMENT, (texture_image->step & 3) ? 1 : 4);
        glPixelStorei(GL_PACK_ROW_LENGTH,
                      GLint(texture_image->step / texture_image->elemSize()));
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        // glBindRenderbuffer(GL_RENDERBUFFER, rbo_texture_);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, image_width_, image_height_, GL_BGRA, GL_UNSIGNED_BYTE,
                     texture_image->data);
        // glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        renderer_geometry_ptr_->DetachContext();
        texture_image_fetched_ = true;
        return true;
    }

    bool TextureRendererCore::FetchNormalImage(cv::Mat *normal_image)
    {
        if (!initial_set_up_ || !image_rendered_)
            return false;
        if (normal_image_fetched_)
            return true;
        renderer_geometry_ptr_->MakeContextCurrent();
        glPixelStorei(GL_PACK_ALIGNMENT, (normal_image->step & 3) ? 1 : 4);
        glPixelStorei(GL_PACK_ROW_LENGTH,
                      GLint(normal_image->step / normal_image->elemSize()));
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        // glBindRenderbuffer(GL_RENDERBUFFER, rbo_normal_);
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        glReadPixels(0, 0, image_width_, image_height_, GL_BGRA, GL_UNSIGNED_BYTE,
                     normal_image->data);
        // glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        renderer_geometry_ptr_->DetachContext();
        normal_image_fetched_ = true;
        return true;
    }

    bool TextureRendererCore::FetchDepthImage(cv::Mat *depth_image)
    {
        if (!initial_set_up_ || !image_rendered_)
            return false;
        if (depth_image_fetched_)
            return true;
        renderer_geometry_ptr_->MakeContextCurrent();
        glPixelStorei(GL_PACK_ALIGNMENT, (depth_image->step & 3) ? 1 : 4);
        glPixelStorei(GL_PACK_ROW_LENGTH,
                      GLint(depth_image->step / depth_image->elemSize()));
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth_);
        glReadPixels(0, 0, image_width_, image_height_, GL_DEPTH_COMPONENT,
                     GL_UNSIGNED_SHORT, depth_image->data);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        renderer_geometry_ptr_->DetachContext();
        depth_image_fetched_ = true;
        return true;
    }

    void TextureRendererCore::CreateBufferObjects()
    {
        renderer_geometry_ptr_->MakeContextCurrent();

        // Initialize renderbuffer bodies_render_data
        glGenRenderbuffers(1, &rbo_texture_);
        glGenRenderbuffers(1, &rbo_normal_);
        glGenRenderbuffers(1, &rbo_depth_);

        // Allocate data storage
        glBindRenderbuffer(GL_RENDERBUFFER, rbo_texture_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, image_width_, image_height_);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo_normal_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, image_width_, image_height_);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, image_width_, image_height_);
        // Detach renderbuffer
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        // Initialize framebuffer bodies_render_data
        glGenFramebuffers(1, &fbo_);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo_texture_);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, rbo_normal_);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth_);

        // Enable draw-buffers
        GLuint draw_buffers[] = {
            GL_COLOR_ATTACHMENT0,
            GL_COLOR_ATTACHMENT1};
        glDrawBuffers(2, draw_buffers);

        // Clean-up
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        renderer_geometry_ptr_->DetachContext();
    }

    void TextureRendererCore::DeleteBufferObjects()
    {
        renderer_geometry_ptr_->MakeContextCurrent();
        glDeleteRenderbuffers(1, &rbo_texture_);
        glDeleteRenderbuffers(1, &rbo_normal_);
        glDeleteRenderbuffers(1, &rbo_depth_);
        glDeleteFramebuffers(1, &fbo_);
        renderer_geometry_ptr_->DetachContext();
    }

    FullTextureRenderer::FullTextureRenderer(
        const std::string &name,
        const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
        const Transform3fA &world2camera_pose, const Intrinsics &intrinsics,
        float z_min, float z_max)
        : FullDepthRenderer{
              name, renderer_geometry_ptr, world2camera_pose, intrinsics, z_min,
              z_max} {}

    FullTextureRenderer::FullTextureRenderer(
        const std::string &name,
        const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
        const std::shared_ptr<Camera> &camera_ptr, float z_min, float z_max)
        : FullDepthRenderer{name, renderer_geometry_ptr, camera_ptr, z_min, z_max}
    {
    }

    FullTextureRenderer::FullTextureRenderer(
        const std::string &name, const std::filesystem::path &metafile_path,
        const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
        const std::shared_ptr<Camera> &camera_ptr)
        : FullDepthRenderer{name, metafile_path, renderer_geometry_ptr,
                            camera_ptr} {}

    bool FullTextureRenderer::SetUp()
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        set_up_ = false;
        if (!metafile_path_.empty())
            if (!LoadMetaData())
                return false;

        // Check if all required objects are set up
        if (!renderer_geometry_ptr_->set_up())
        {
            std::cerr << "Renderer geometry " << renderer_geometry_ptr_->name()
                      << " was not set up" << std::endl;
            return false;
        }
        if (camera_ptr_ && !InitParametersFromCamera())
            return false;

        // Set up everything
        CalculateProjectionMatrix();
        CalculateProjectionTerms();
        ClearDepthImage();
        ClearNormalImage();
        ClearTextureImage();
        if (!core_.SetUp(renderer_geometry_ptr_, intrinsics_.width,
                         intrinsics_.height))
            return false;

        set_up_ = true;
        return true;
    }

    bool FullTextureRenderer::StartRendering()
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        if (!set_up_)
        {
            std::cerr << "Set up renderer " << name_ << " first" << std::endl;
            return false;
        }
        return core_.StartRendering(projection_matrix_, world2camera_pose_);
    }

    bool FullTextureRenderer::FetchTextureImage()
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        if (!set_up_)
        {
            std::cerr << "Set up renderer " << name_ << " first" << std::endl;
            return false;
        }
        return core_.FetchTextureImage(&texture_image_);
    }

    bool FullTextureRenderer::FetchNormalImage()
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        if (!set_up_)
        {
            std::cerr << "Set up renderer " << name_ << " first" << std::endl;
            return false;
        }
        return core_.FetchNormalImage(&normal_image_);
    }

    bool FullTextureRenderer::FetchDepthImage()
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        if (!set_up_)
        {
            std::cerr << "Set up renderer " << name_ << " first" << std::endl;
            return false;
        }
        return core_.FetchDepthImage(&depth_image_);
    }

    const cv::Mat &FullTextureRenderer::texture_image() const
    {
        return texture_image_;
    }

    const cv::Mat &FullTextureRenderer::normal_image() const
    {
        return normal_image_;
    }

    Eigen::Vector3f FullTextureRenderer::NormalVector(
        cv::Vec4b normal_image_value) const
    {
        return Eigen::Vector3f{1.0f - float(normal_image_value[0]) / 127.5f,
                               1.0f - float(normal_image_value[1]) / 127.5f,
                               1.0f - float(normal_image_value[2]) / 127.5f};
    }

    Eigen::Vector3f FullTextureRenderer::NormalVector(
        const cv::Point2i &image_coordinate) const
    {
        auto normal_image_value{normal_image_.at<cv::Vec4b>(image_coordinate)};
        return Eigen::Vector3f{1.0f - float(normal_image_value[0]) / 127.5f,
                               1.0f - float(normal_image_value[1]) / 127.5f,
                               1.0f - float(normal_image_value[2]) / 127.5f};
    }

    cv::Vec4b FullTextureRenderer::NormalImageValue(
        const cv::Point2i &image_coordinate) const
    {
        return normal_image_.at<cv::Vec4b>(image_coordinate);
    }

    bool FullTextureRenderer::LoadMetaData()
    {
        // Open file storage from yaml
        cv::FileStorage fs;
        if (!OpenYamlFileStorage(metafile_path_, &fs))
            return false;

        // Read parameters from yaml
        ReadOptionalValueFromYaml(fs, "z_min", &z_min_);
        ReadOptionalValueFromYaml(fs, "z_max", &z_max_);
        fs.release();
        return true;
    }

    void FullTextureRenderer::ClearNormalImage()
    {
        normal_image_.create(cv::Size{intrinsics_.width, intrinsics_.height},
                             CV_8UC4);
        normal_image_.setTo(cv::Vec4b{0, 0, 0, 0});
    }

    void FullTextureRenderer::ClearTextureImage()
    {
        texture_image_.create(cv::Size{intrinsics_.width, intrinsics_.height},
                              CV_8UC4);
        texture_image_.setTo(cv::Vec4b{0, 0, 0, 0});
    }

    FocusedTextureRenderer::FocusedTextureRenderer(
        const std::string &name,
        const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
        const Transform3fA &world2camera_pose, const Intrinsics &intrinsics,
        int image_size, float z_min, float z_max)
        : FocusedDepthRenderer{name,
                               renderer_geometry_ptr,
                               world2camera_pose,
                               intrinsics,
                               image_size,
                               z_min,
                               z_max} {}

    FocusedTextureRenderer::FocusedTextureRenderer(
        const std::string &name,
        const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
        const std::shared_ptr<Camera> &camera_ptr, int image_size, float z_min,
        float z_max)
        : FocusedDepthRenderer{
              name, renderer_geometry_ptr, camera_ptr, image_size, z_min, z_max} {}

    FocusedTextureRenderer::FocusedTextureRenderer(
        const std::string &name, const std::filesystem::path &metafile_path,
        const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
        const std::shared_ptr<Camera> &camera_ptr)
        : FocusedDepthRenderer{name, metafile_path, renderer_geometry_ptr,
                               camera_ptr} {}

    bool FocusedTextureRenderer::SetUp()
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        set_up_ = false;
        if (!metafile_path_.empty())
            if (!LoadMetaData())
                return false;

        // Check if all required objects are set up
        if (!renderer_geometry_ptr_->set_up())
        {
            std::cerr << "Renderer geometry " << renderer_geometry_ptr_->name()
                      << " was not set up" << std::endl;
            return false;
        }
        if (camera_ptr_ && !InitParametersFromCamera())
            return false;
        if (referenced_body_ptrs_.empty())
        {
            std::cerr << "No referenced bodys were assigned to renderer " << name_
                      << std::endl;
            return false;
        }
        for (auto &referenced_body_ptr : referenced_body_ptrs_)
        {
            if (!referenced_body_ptr->set_up())
            {
                std::cerr << "Body " << referenced_body_ptr->name() << " was not set up"
                          << std::endl;
                return false;
            }
        }

        // Set up everything
        CalculateProjectionMatrix();
        CalculateProjectionTerms();
        ClearDepthImage();
        ClearNormalImage();
        ClearTextureImage();
        if (!core_.SetUp(renderer_geometry_ptr_, image_size_, image_size_))
            return false;

        set_up_ = true;
        return true;
    }

    bool FocusedTextureRenderer::StartRendering()
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        if (!set_up_)
        {
            std::cerr << "Set up renderer " << name_ << " first" << std::endl;
            return false;
        }
        CalculateProjectionMatrix();
        return core_.StartRendering(projection_matrix_, world2camera_pose_);
    }

    bool FocusedTextureRenderer::FetchTextureImage()
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        if (!set_up_)
        {
            std::cerr << "Set up renderer " << name_ << " first" << std::endl;
            return false;
        }
        return core_.FetchTextureImage(&focused_texture_image_);
    }

    bool FocusedTextureRenderer::FetchNormalImage()
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        if (!set_up_)
        {
            std::cerr << "Set up renderer " << name_ << " first" << std::endl;
            return false;
        }
        return core_.FetchNormalImage(&focused_normal_image_);
    }

    bool FocusedTextureRenderer::FetchDepthImage()
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        if (!set_up_)
        {
            std::cerr << "Set up renderer " << name_ << " first" << std::endl;
            return false;
        }
        return core_.FetchDepthImage(&focused_depth_image_);
    }

    const cv::Mat &FocusedTextureRenderer::focused_texture_image() const
    {
        return focused_texture_image_;
    }

    const cv::Mat &FocusedTextureRenderer::focused_normal_image() const
    {
        return focused_normal_image_;
    }

    Eigen::Vector3f FocusedTextureRenderer::NormalVector(
        cv::Vec4b normal_image_value) const
    {
        return Eigen::Vector3f{1.0f - float(normal_image_value[0]) / 127.5f,
                               1.0f - float(normal_image_value[1]) / 127.5f,
                               1.0f - float(normal_image_value[2]) / 127.5f};
    }

    Eigen::Vector3f FocusedTextureRenderer::NormalVector(
        const cv::Point2i &image_coordinate) const
    {
        int u = int((image_coordinate.x - corner_u_) * scale_ + 0.5f);
        int v = int((image_coordinate.y - corner_v_) * scale_ + 0.5f);
        auto normal_image_value{focused_normal_image_.at<cv::Vec4b>(v, u)};
        return Eigen::Vector3f{1.0f - float(normal_image_value[0]) / 127.5f,
                               1.0f - float(normal_image_value[1]) / 127.5f,
                               1.0f - float(normal_image_value[2]) / 127.5f};
    }

    cv::Vec4b FocusedTextureRenderer::NormalImageValue(
        const cv::Point2i &image_coordinate) const
    {
        int u = int((image_coordinate.x - corner_u_) * scale_ + 0.5f);
        int v = int((image_coordinate.y - corner_v_) * scale_ + 0.5f);
        return focused_normal_image_.at<cv::Vec4b>(v, u);
    }

    bool FocusedTextureRenderer::LoadMetaData()
    {
        // Open file storage from yaml
        cv::FileStorage fs;
        if (!OpenYamlFileStorage(metafile_path_, &fs))
            return false;

        // Read parameters from yaml
        ReadOptionalValueFromYaml(fs, "z_min", &z_min_);
        ReadOptionalValueFromYaml(fs, "z_max", &z_max_);
        ReadOptionalValueFromYaml(fs, "image_size", &image_size_);
        fs.release();
        return true;
    }

    void FocusedTextureRenderer::ClearTextureImage()
    {
        focused_texture_image_.create(cv::Size{image_size_, image_size_}, CV_8UC4);
        focused_texture_image_.setTo(cv::Vec4b{0, 0, 0, 0});
    }

    void FocusedTextureRenderer::ClearNormalImage()
    {
        focused_normal_image_.create(cv::Size{image_size_, image_size_}, CV_8UC4);
        focused_normal_image_.setTo(cv::Vec4b{0, 0, 0, 0});
    }

} // namespace icg
