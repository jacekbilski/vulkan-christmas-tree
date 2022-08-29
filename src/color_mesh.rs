use ash::vk;
use cgmath::{Matrix4, SquareMatrix};
use memoffset::offset_of;

use crate::vulkan::Vertex;

#[derive(Debug)]
pub struct ColorMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub instances: Vec<InstanceData>,
}

#[repr(C)]
#[derive(Debug)]
pub struct InstanceData {
    pub model: Matrix4<f32>,
    pub color: Color,
    pub padding: [f32; 2], // needed for std430 layout
}

impl InstanceData {
    pub fn get_binding_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription {
            binding: 1,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::INSTANCE,
        }]
    }

    pub fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        let matrix_quarter = (std::mem::size_of::<Matrix4<f32>>() / 4) as u32;
        let color_part = (std::mem::size_of::<[f32; 3]>()) as u32;
        vec![
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 2,
                format: vk::Format::R32G32B32_SFLOAT, // aka vec3
                offset: offset_of!(Self, color) as u32 + 0 * color_part,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 3,
                format: vk::Format::R32G32B32_SFLOAT, // aka vec3
                offset: offset_of!(Self, color) as u32 + 1 * color_part,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 4,
                format: vk::Format::R32G32B32_SFLOAT, // aka vec3
                offset: offset_of!(Self, color) as u32 + 2 * color_part,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 5,
                format: vk::Format::R32_SFLOAT, // aka float
                offset: offset_of!(Self, color) as u32 + 3 * color_part,
            },
            // need four because I'm sending a 4x4 matrix
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 6,
                format: vk::Format::R32G32B32A32_SFLOAT, // aka vec4
                offset: offset_of!(Self, model) as u32 + 0 * matrix_quarter,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 7,
                format: vk::Format::R32G32B32A32_SFLOAT, // aka vec4
                offset: offset_of!(Self, model) as u32 + 1 * matrix_quarter,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 8,
                format: vk::Format::R32G32B32A32_SFLOAT, // aka vec4
                offset: offset_of!(Self, model) as u32 + 2 * matrix_quarter,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 9,
                format: vk::Format::R32G32B32A32_SFLOAT, // aka vec4
                offset: offset_of!(Self, model) as u32 + 3 * matrix_quarter,
            },
        ]
    }
}

impl Default for InstanceData {
    fn default() -> Self {
        Self {
            model: Matrix4::identity(),
            color: Color::default(),
            padding: [0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Color {
    pub ambient: [f32; 3],
    pub diffuse: [f32; 3],
    pub specular: [f32; 3],
    pub shininess: f32,
}

impl Default for Color {
    fn default() -> Self {
        Self {
            ambient: [0.0, 0.0, 0.0],
            diffuse: [0.0, 0.0, 0.0],
            specular: [0.0, 0.0, 0.0],
            shininess: 0.0,
        }
    }
}
