// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Manuel Stoiber, German Aerospace Center (DLR)

#include <icg/body.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader/tiny_obj_loader.h>

#define STB_IMAGE_IMPLEMENTATION
#include <icg/stb_image.h>

namespace icg
{

  Body::Body(const std::string &name, const std::filesystem::path &geometry_path,
             float geometry_unit_in_meter, bool geometry_counterclockwise,
             bool geometry_enable_culling, const Transform3fA &geometry2body_pose,
             uchar silhouette_id)
      : name_{name},
        geometry_path_{geometry_path},
        geometry_unit_in_meter_{geometry_unit_in_meter},
        geometry_counterclockwise_{geometry_counterclockwise},
        geometry_enable_culling_{geometry_enable_culling},
        geometry2body_pose_{geometry2body_pose},
        silhouette_id_{silhouette_id}
  {
    geometry2world_pose_ = geometry2body_pose;
    world2geometry_pose_ = geometry2world_pose_.inverse();
  }

  Body::Body(const std::string &name, const std::filesystem::path &metafile_path)
      : name_{name}, metafile_path_{metafile_path} {}

  bool Body::SetUp()
  {
    set_up_ = false;
    if (!metafile_path_.empty())
      if (!LoadMetaData())
        return false;
    if (!LoadMeshData())
      return false;
    if (!CalculateMaximumBodyDiameter())
      return false;
    set_up_ = true;
    return true;
  }

  Body::~Body()
  {
    // FIXME: do I have memory leaks here?
    if (texture_loaded_)
    {
      // stbi_image_free(texture_data_);
      // texture_loaded_ = false;
    }
  }

  void Body::set_name(const std::string &name) { name_ = name; }

  void Body::set_metafile_path(const std::filesystem::path &metafile_path)
  {
    metafile_path_ = metafile_path;
    set_up_ = false;
  }

  void Body::set_geometry_path(const std::filesystem::path &geometry_path)
  {
    geometry_path_ = geometry_path;
    set_up_ = false;
  }

  void Body::set_geometry_unit_in_meter(float geometry_unit_in_meter)
  {
    geometry_unit_in_meter_ = geometry_unit_in_meter;
    set_up_ = false;
  }

  void Body::set_geometry_counterclockwise(bool geometry_counterclockwise)
  {
    geometry_counterclockwise_ = geometry_counterclockwise;
  }

  void Body::set_geometry_enable_culling(bool geometry_enable_culling)
  {
    geometry_enable_culling_ = geometry_enable_culling;
  }

  void Body::set_geometry2body_pose(const Transform3fA &geometry2body_pose)
  {
    geometry2body_pose_ = geometry2body_pose;
    geometry2world_pose_ = body2world_pose_ * geometry2body_pose_;
    world2geometry_pose_ = geometry2world_pose_.inverse();
    set_up_ = false;
  }

  bool Body::set_silhouette_id(uchar silhouette_id)
  {
    silhouette_id_ = silhouette_id;
    return true;
  }

  void Body::set_body2world_pose(const Transform3fA &body2world_pose)
  {
    body2world_pose_ = body2world_pose;
    world2body_pose_ = body2world_pose_.inverse();
    geometry2world_pose_ = body2world_pose_ * geometry2body_pose_;
    world2geometry_pose_ = geometry2world_pose_.inverse();
  }

  void Body::set_world2body_pose(const Transform3fA &world2body_pose)
  {
    world2body_pose_ = world2body_pose;
    body2world_pose_ = world2body_pose_.inverse();
    geometry2world_pose_ = body2world_pose_ * geometry2body_pose_;
    world2geometry_pose_ = geometry2world_pose_.inverse();
  }

  const std::string &Body::name() const { return name_; }

  const std::filesystem::path &Body::metafile_path() const
  {
    return metafile_path_;
  }

  const std::filesystem::path &Body::geometry_path() const
  {
    return geometry_path_;
  }

  float Body::geometry_unit_in_meter() const { return geometry_unit_in_meter_; }

  bool Body::geometry_counterclockwise() const
  {
    return geometry_counterclockwise_;
  }

  bool Body::geometry_enable_culling() const { return geometry_enable_culling_; }

  const Transform3fA &Body::geometry2body_pose() const
  {
    return geometry2body_pose_;
  }

  uchar Body::silhouette_id() const { return silhouette_id_; }

  const Transform3fA &Body::body2world_pose() const { return body2world_pose_; }

  const Transform3fA &Body::world2body_pose() const { return world2body_pose_; }

  const Transform3fA &Body::geometry2world_pose() const
  {
    return geometry2world_pose_;
  }

  const Transform3fA &Body::world2geometry_pose() const
  {
    return world2geometry_pose_;
  }

  const std::vector<std::array<int, 3>> &Body::mesh_vertex_indices() const
  {
    return mesh_vertex_indices_;
  }

  const std::vector<std::array<int, 3>> &Body::mesh_texture_indices() const
  {
    return mesh_texture_indices_;
  }

  const std::vector<Eigen::Vector3f> &Body::vertices() const { return vertices_; }

  const std::vector<std::array<float, 2>> &Body::texture_coords() const { return texture_coords_; }

  float Body::maximum_body_diameter() const { return maximum_body_diameter_; }

  std::string Body::texture_path() const { return texture_path_; }

  unsigned char *Body::texture_data() const { return texture_data_; }

  int Body::texture_width() const { return texture_width_; }

  int Body::texture_height() const { return texture_height_; }

  int Body::texture_channels() const { return texture_channels_; }

  bool Body::set_up() const { return set_up_; }

  bool Body::enable_texture() const { return enable_texture_; }

  bool Body::LoadMetaData()
  {
    // Open file storage from yaml
    cv::FileStorage fs;
    if (!OpenYamlFileStorage(metafile_path_, &fs))
      return false;

    // Read parameters from yaml
    if (!(ReadRequiredValueFromYaml(fs, "geometry_path", &geometry_path_) &&
          ReadRequiredValueFromYaml(fs, "geometry_unit_in_meter",
                                    &geometry_unit_in_meter_) &&
          ReadRequiredValueFromYaml(fs, "geometry_counterclockwise",
                                    &geometry_counterclockwise_) &&
          ReadRequiredValueFromYaml(fs, "geometry_enable_culling",
                                    &geometry_enable_culling_) &&
          ReadRequiredValueFromYaml(fs, "geometry_enable_texture",
                                    &enable_texture_) &&
          ReadRequiredValueFromYaml(fs, "geometry2body_pose",
                                    &geometry2body_pose_)))
    {
      std::cerr << "Could not read all required body parameters from "
                << metafile_path_ << std::endl;
      return false;
    }
    ReadOptionalValueFromYaml(fs, "silhouette_id", &silhouette_id_);
    fs.release();

    // Process parameters
    if (geometry_path_ == "INFER_FROM_NAME")
      geometry_path_ = metafile_path_.parent_path() / (name_ + ".obj");
    else if (geometry_path_.is_relative())
      geometry_path_ = metafile_path_.parent_path() / geometry_path_;
    geometry2world_pose_ = body2world_pose_ * geometry2body_pose_;
    world2geometry_pose_ = geometry2world_pose_.inverse();
    return true;
  }

  bool Body::LoadMeshData()
  {
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warning;
    std::string error;
    if (enable_texture_)
    {
      // Load obj file with texture
      if (!tinyobj::LoadObj(&attributes, &shapes, &materials, &warning, &error,
                            geometry_path_.string().c_str(), geometry_path_.parent_path().string().c_str(), true,
                            false))
      {
        std::cerr << "TinyObjLoader failed to load data from " << geometry_path_
                  << std::endl;
        return false;
      }
    }
    else
    {
      // Load obj file without texture
      if (!tinyobj::LoadObj(&attributes, &shapes, &materials, &warning, &error,
                            geometry_path_.string().c_str(), nullptr, true,
                            false))
      {
        std::cerr << "TinyObjLoader failed to load data from " << geometry_path_
                  << std::endl;
        return false;
      }
    }

    if (!error.empty())
      std::cerr << error << std::endl;

    // Load vertices and scale them if needed
    vertices_.resize(attributes.vertices.size() / 3);
    memcpy(vertices_.data(), attributes.vertices.data(),
           sizeof(float) * attributes.vertices.size());
    if (geometry_unit_in_meter_ != 1.0f)
    {
      for (auto &vertex : vertices_)
      {
        vertex *= geometry_unit_in_meter_;
      }
    }

    // Load texture coordinates
    if (enable_texture_)
    {
      texture_coords_.resize(attributes.texcoords.size() / 2);
      memcpy(texture_coords_.data(), attributes.texcoords.data(),
             sizeof(float) * attributes.texcoords.size());
    }

    // Load material
    if (enable_texture_)
    {
      if (materials.size() != 1)
      {
        std::cerr << "Mesh contains more than one material" << std::endl;
        return false;
      }
      // Load texture data
      texture_path_ = geometry_path_.parent_path() / materials[0].diffuse_texname;
      stbi_set_flip_vertically_on_load(true);  // OpenGL expects lower left corner
      texture_data_ = stbi_load(texture_path_.c_str(), &texture_width_, &texture_height_, &texture_channels_, 0);
      texture_loaded_ = true;
    }

    // Load mesh vertices
    mesh_vertex_indices_.clear();
    mesh_texture_indices_.clear();
    for (const auto &shape : shapes)
    {
      size_t index_offset = 0;
      for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f)
      {
        if (shape.mesh.num_face_vertices[f] != 3)
        {
          std::cerr << "Mesh contains non triangle shapes" << std::endl;
          index_offset += shape.mesh.num_face_vertices[f];
          continue;
        }

        if (geometry_counterclockwise_)
        {
          mesh_vertex_indices_.push_back(std::array<int, 3>{
              shape.mesh.indices[index_offset].vertex_index,
              shape.mesh.indices[index_offset + 1].vertex_index,
              shape.mesh.indices[index_offset + 2].vertex_index});
          if (enable_texture_)
          {
            mesh_texture_indices_.push_back(std::array<int, 3>{
                shape.mesh.indices[index_offset].texcoord_index,
                shape.mesh.indices[index_offset + 1].texcoord_index,
                shape.mesh.indices[index_offset + 2].texcoord_index});
          }
        }
        else
        {
          mesh_vertex_indices_.push_back(std::array<int, 3>{
              shape.mesh.indices[index_offset + 2].vertex_index,
              shape.mesh.indices[index_offset + 1].vertex_index,
              shape.mesh.indices[index_offset].vertex_index});
          if (enable_texture_)
          {
            mesh_texture_indices_.push_back(std::array<int, 3>{
                shape.mesh.indices[index_offset + 2].texcoord_index,
                shape.mesh.indices[index_offset + 1].texcoord_index,
                shape.mesh.indices[index_offset].texcoord_index});
          }
        }
        index_offset += 3;
      }
    }
    return true;
  }

  bool Body::CalculateMaximumBodyDiameter()
  {
    float max_radius = 0.0f;
    for (const auto &vertex : vertices_)
    {
      max_radius = std::max(max_radius, (geometry2body_pose_ * vertex).norm());
    }
    maximum_body_diameter_ = 2.0f * max_radius;
    return true;
  }

} // namespace icg
