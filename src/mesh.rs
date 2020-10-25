use ash::vk;
use cgmath::{Matrix4, SquareMatrix};
use memoffset::offset_of;

use crate::vulkan::Vertex;

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub instances: Vec<InstanceData>,
}

#[repr(C)]
pub struct InstanceData {
    pub model: Matrix4<f32>,
}
impl InstanceData {
    pub fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 1,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::INSTANCE,
        }]
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4] {
        let quarter = (std::mem::size_of::<Self>() / 4) as u32;
        [
            // need four because I'm sending a 4x4 matrix
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 2,
                format: vk::Format::R32G32B32A32_SFLOAT, // aka vec4
                offset: offset_of!(Self, model) as u32 + 0 * quarter,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 3,
                format: vk::Format::R32G32B32A32_SFLOAT, // aka vec4
                offset: offset_of!(Self, model) as u32 + 1 * quarter,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 4,
                format: vk::Format::R32G32B32A32_SFLOAT, // aka vec4
                offset: offset_of!(Self, model) as u32 + 2 * quarter,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 5,
                format: vk::Format::R32G32B32A32_SFLOAT, // aka vec4
                offset: offset_of!(Self, model) as u32 + 3 * quarter,
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
