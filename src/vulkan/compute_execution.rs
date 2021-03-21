use std::ptr;

use ash::vk;

use crate::scene::snow::{Snowflake, MAX_SNOWFLAKES};
use crate::vulkan::compute_setup::VulkanComputeSetup;
use crate::vulkan::core::VulkanCore;

const WORKGROUP_SIZE: u32 = 64;

pub struct VulkanComputeExecution {
    core: VulkanCore,
    compute_setup: VulkanComputeSetup,

    descriptor_set: vk::DescriptorSet,
    command_buffer: vk::CommandBuffer,

    snowflake_positions_buffer: vk::Buffer,
    snowflake_positions_buffer_memory: vk::DeviceMemory,

    snowflake_velocities_buffer: vk::Buffer,
    snowflake_velocities_buffer_memory: vk::DeviceMemory,

    snowflake_accelerations_buffer: vk::Buffer,
    snowflake_accelerations_buffer_memory: vk::DeviceMemory,

    fence: vk::Fence,
}

impl VulkanComputeExecution {
    pub fn new(
        core: VulkanCore,
        compute_setup: VulkanComputeSetup,
        snowflakes: &Vec<Snowflake>,
        drawing_buffer: vk::Buffer,
        drawing_buffer_size: usize,
    ) -> Self {
        let (snowflake_positions_buffer, snowflake_positions_buffer_memory) = core
            .create_data_buffer(
                compute_setup.command_pool,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                snowflakes,
            );
        let snowflakes_buffer_size = std::mem::size_of::<Snowflake>() * snowflakes.len();
        let zero_vec = vec![0u8; snowflakes_buffer_size];
        let (snowflake_velocities_buffer, snowflake_velocities_buffer_memory) = core
            .create_data_buffer(
                compute_setup.command_pool,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                &zero_vec,
            );
        let (snowflake_accelerations_buffer, snowflake_accelerations_buffer_memory) = core
            .create_data_buffer(
                compute_setup.command_pool,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                &zero_vec,
            );
        let descriptor_set = VulkanComputeExecution::create_descriptor_set(
            &core.device,
            compute_setup.descriptor_pool,
            compute_setup.descriptor_set_layout,
            snowflakes_buffer_size,
            snowflake_positions_buffer,
            snowflake_velocities_buffer,
            snowflake_accelerations_buffer,
            drawing_buffer,
            drawing_buffer_size,
        );
        let command_buffer = VulkanComputeExecution::create_command_buffer(
            &core,
            &compute_setup,
            descriptor_set,
            0,
            0.0,
        );
        let fence = core.create_fence();

        VulkanComputeExecution {
            core,
            compute_setup,

            descriptor_set,
            command_buffer,

            snowflake_positions_buffer,
            snowflake_positions_buffer_memory,

            snowflake_velocities_buffer,
            snowflake_velocities_buffer_memory,

            snowflake_accelerations_buffer,
            snowflake_accelerations_buffer_memory,

            fence,
        }
    }

    fn create_descriptor_set(
        device: &ash::Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        snowflakes_buffer_size: usize,
        snowflake_positions_buffer: vk::Buffer,
        snowflake_velocities_buffer: vk::Buffer,
        snowflake_accelerations_buffer: vk::Buffer,
        drawing_buffer: vk::Buffer,
        drawing_buffer_size: usize,
    ) -> vk::DescriptorSet {
        let descriptor_set_layouts = [descriptor_set_layout];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: 1 as u32,
            p_set_layouts: descriptor_set_layouts.as_ptr(),
            ..Default::default()
        };

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets!")
        };

        for &descritptor_set in descriptor_sets.iter() {
            let descriptor_buffer_info = [
                vk::DescriptorBufferInfo {
                    buffer: snowflake_positions_buffer,
                    offset: 0,
                    range: snowflakes_buffer_size as u64,
                },
                vk::DescriptorBufferInfo {
                    buffer: snowflake_velocities_buffer,
                    offset: 0,
                    range: snowflakes_buffer_size as u64,
                },
                vk::DescriptorBufferInfo {
                    buffer: snowflake_accelerations_buffer,
                    offset: 0,
                    range: snowflakes_buffer_size as u64,
                },
                vk::DescriptorBufferInfo {
                    buffer: drawing_buffer,
                    offset: 0,
                    range: drawing_buffer_size as u64,
                },
            ];

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

    fn create_command_buffer(
        core: &VulkanCore,
        compute_setup: &VulkanComputeSetup,
        descriptor_set: vk::DescriptorSet,
        frame_no: u32,
        last_frame_time_secs: f32,
    ) -> vk::CommandBuffer {
        let device = &core.device;
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            command_buffer_count: 1,
            command_pool: compute_setup.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };

        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")[0]
        };

        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            p_inheritance_info: ptr::null(),
            flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            ..Default::default()
        };

        unsafe {
            device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!");

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                compute_setup.pipeline,
            );

            let descriptor_sets_to_bind = [descriptor_set];
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                compute_setup.pipeline_layout,
                0,
                &descriptor_sets_to_bind,
                &[],
            );

            device.cmd_push_constants(
                command_buffer,
                compute_setup.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &frame_no.to_le_bytes(),
            );
            device.cmd_push_constants(
                command_buffer,
                compute_setup.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                4,
                &last_frame_time_secs.to_le_bytes(),
            );

            device.cmd_dispatch(
                command_buffer,
                (MAX_SNOWFLAKES as f32 / WORKGROUP_SIZE as f32).ceil() as u32,
                1,
                1,
            );

            device
                .end_command_buffer(command_buffer)
                .expect("Failed to record Command Buffer at Ending!");
        }

        command_buffer
    }

    pub fn do_calculations(
        &mut self,
        snow_calculated_semaphore: vk::Semaphore,
        frame_no: u32,
        last_frame_time_secs: f32,
    ) {
        let new_command_buffer = VulkanComputeExecution::create_command_buffer(
            &self.core,
            &self.compute_setup,
            self.descriptor_set,
            frame_no,
            last_frame_time_secs,
        );
        let command_buffers = [new_command_buffer];

        let wait_fences = [self.fence];
        unsafe {
            self.core
                .device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence!");

            // only now I'm sure it's not used any more
            self.core
                .device
                .free_command_buffers(self.compute_setup.command_pool, &vec![self.command_buffer]);
        }

        self.command_buffer = new_command_buffer;

        let wait_stages = [vk::PipelineStageFlags::COMPUTE_SHADER];
        let signal_semaphores = [snow_calculated_semaphore];
        let submit_infos = [vk::SubmitInfo {
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: command_buffers.len() as u32,
            p_command_buffers: command_buffers.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        }];
        unsafe {
            self.core
                .device
                .reset_fences(&wait_fences)
                .expect("Failed to reset Fence!");

            self.core
                .device
                .queue_submit(self.core.compute_queue, &submit_infos, self.fence)
                .expect("Failed to execute queue submit.");
        }
    }

    pub fn drop(&self, compute_setup: &VulkanComputeSetup) {
        unsafe {
            let device = &self.core.device;
            device.destroy_buffer(self.snowflake_accelerations_buffer, None);
            device.free_memory(self.snowflake_accelerations_buffer_memory, None);
            device.destroy_buffer(self.snowflake_velocities_buffer, None);
            device.free_memory(self.snowflake_velocities_buffer_memory, None);
            device.destroy_buffer(self.snowflake_positions_buffer, None);
            device.free_memory(self.snowflake_positions_buffer_memory, None);
            device.free_command_buffers(compute_setup.command_pool, &vec![self.command_buffer]);
            device.destroy_fence(self.fence, None);
        }
    }
}
