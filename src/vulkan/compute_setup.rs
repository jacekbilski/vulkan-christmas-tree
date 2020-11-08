use std::ffi::CString;

use ash::version::DeviceV1_0;
use ash::vk;

use crate::vulkan::core::VulkanCore;

pub struct VulkanComputeSetup {
    core: VulkanCore,

    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,

    pub command_pool: vk::CommandPool,
    pub descriptor_pool: vk::DescriptorPool,
}

impl VulkanComputeSetup {
    pub fn new(core: VulkanCore) -> Self {
        let descriptor_set_layout = VulkanComputeSetup::create_descriptor_set_layout(&core.device);
        let (pipeline, pipeline_layout) =
            VulkanComputeSetup::create_pipeline(&core, descriptor_set_layout);
        let command_pool = core.create_command_pool(core.queue_family.compute_family.unwrap());
        let descriptor_pool = VulkanComputeSetup::create_descriptor_pool(&core.device);

        VulkanComputeSetup {
            core,

            descriptor_set_layout,
            pipeline_layout,
            pipeline,

            command_pool,
            descriptor_pool,
        }
    }

    fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
        let descriptor_set_layout_bindings = [vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        }];

        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: descriptor_set_layout_bindings.len() as u32,
            p_bindings: descriptor_set_layout_bindings.as_ptr(),
            ..Default::default()
        };

        unsafe {
            device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .expect("Failed to create Descriptor Set Layout!")
        }
    }

    fn create_pipeline(
        core: &VulkanCore,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let comp_shader_module = core.create_shader_module("simple.comp.spv");

        let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

        let shader_stages = vk::PipelineShaderStageCreateInfo {
            module: comp_shader_module,
            p_name: main_function_name.as_ptr(),
            stage: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        };

        let set_layouts = [descriptor_set_layout];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            core.device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Failed to create pipeline layout!")
        };

        let compute_pipeline_create_infos = [vk::ComputePipelineCreateInfo {
            flags: vk::PipelineCreateFlags::empty(),
            layout: pipeline_layout,
            stage: shader_stages,
            ..Default::default()
        }];

        let compute_pipelines = unsafe {
            core.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &compute_pipeline_create_infos,
                    None,
                )
                .expect("Failed to create Compute Pipeline!.")
        };

        unsafe {
            core.device.destroy_shader_module(comp_shader_module, None);
        }

        (compute_pipelines[0], pipeline_layout)
    }

    fn create_descriptor_pool(device: &ash::Device) -> vk::DescriptorPool {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        }];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };

        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create Descriptor Pool!")
        }
    }

    pub fn drop(&self) {
        unsafe {
            let device = &self.core.device;
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
