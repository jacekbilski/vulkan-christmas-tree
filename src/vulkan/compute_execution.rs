use std::ptr;

use ash::version::DeviceV1_0;
use ash::vk;

use crate::vulkan::compute_setup::VulkanComputeSetup;
use crate::vulkan::core::VulkanCore;

pub struct VulkanComputeExecution {
    core: VulkanCore,

    descriptor_set: vk::DescriptorSet,
}

impl VulkanComputeExecution {
    pub fn new(
        core: VulkanCore,
        compute_setup: &VulkanComputeSetup,
        buffer: vk::Buffer,
        buffer_size: usize,
    ) -> Self {
        let descriptor_set = VulkanComputeExecution::create_descriptor_set(
            &core.device,
            compute_setup.descriptor_pool,
            compute_setup.descriptor_set_layout,
            buffer,
            buffer_size,
        );

        VulkanComputeExecution {
            core,

            descriptor_set,
        }
    }

    fn create_descriptor_set(
        device: &ash::Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        buffer: vk::Buffer,
        buffer_size: usize,
    ) -> vk::DescriptorSet {
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: 1 as u32,
            p_set_layouts: &descriptor_set_layout,
            ..Default::default()
        };

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets!")
        };

        for &descritptor_set in descriptor_sets.iter() {
            let descriptor_buffer_info = [vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: buffer_size as u64,
            }];

            let descriptor_write_sets = [vk::WriteDescriptorSet {
                dst_set: descritptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: descriptor_buffer_info.len() as u32,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: descriptor_buffer_info.as_ptr(),
                p_texel_buffer_view: ptr::null(),
                ..Default::default()
            }];

            unsafe {
                device.update_descriptor_sets(&descriptor_write_sets, &[]);
            }
        }

        descriptor_sets[0]
    }

    pub fn drop(&self) {
        // nothing yet
    }
}
