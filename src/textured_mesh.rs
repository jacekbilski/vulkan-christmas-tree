use ash::vk;
use cgmath::{Matrix4, SquareMatrix};
use memoffset::offset_of;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TexturedVertex {
    pub pos: [f32; 3],
    pub norm: [f32; 3],
    pub texture_coordinates: [f32; 2],
}

impl TexturedVertex {
    pub fn get_binding_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    pub fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT, // aka vec3
                offset: offset_of!(Self, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT, // aka vec3
                offset: offset_of!(Self, norm) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32_SFLOAT, // aka vec2
                offset: offset_of!(Self, texture_coordinates) as u32,
            },
        ]
    }
}

#[derive(Debug)]
pub struct TexturedMesh {
    pub vertices: Vec<TexturedVertex>,
    pub indices: Vec<u32>,
    pub instances: Vec<InstanceData>,
}

#[repr(C)]
#[derive(Debug)]
pub struct InstanceData {
    pub model: Matrix4<f32>,
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
        vec![
            // need four because I'm sending a 4x4 matrix
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 3,
                format: vk::Format::R32G32B32A32_SFLOAT, // aka vec4
                offset: offset_of!(Self, model) as u32 + 0 * matrix_quarter,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 4,
                format: vk::Format::R32G32B32A32_SFLOAT, // aka vec4
                offset: offset_of!(Self, model) as u32 + 1 * matrix_quarter,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 5,
                format: vk::Format::R32G32B32A32_SFLOAT, // aka vec4
                offset: offset_of!(Self, model) as u32 + 2 * matrix_quarter,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 6,
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
        }
    }
}
