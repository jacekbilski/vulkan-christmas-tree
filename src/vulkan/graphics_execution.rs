use std::ptr;

use ash::version::DeviceV1_0;
use ash::vk;
use cgmath::{Matrix4, Point3};

use crate::mesh::Mesh;
use crate::scene::camera::Camera;
use crate::scene::lights::{Light, Lights};
use crate::vulkan::core::VulkanCore;
use crate::vulkan::graphics_setup::{VulkanGraphicsSetup, CAMERA_UBO_INDEX, LIGHTS_UBO_INDEX};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct UniformBuffer {
    buffers: Vec<vk::Buffer>,              // one per swapchain_image_count
    buffers_memory: Vec<vk::DeviceMemory>, // one per swapchain_image_count
}

#[derive(Clone, Copy)]
struct VulkanMesh {
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    indices_no: u32,
    instance_buffer: vk::Buffer,
    instance_buffer_memory: vk::DeviceMemory,
    instances_no: u32,
}

impl VulkanMesh {
    fn drop(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.instance_buffer, None);
            device.free_memory(self.instance_buffer_memory, None);
            device.destroy_buffer(self.index_buffer, None);
            device.free_memory(self.index_buffer_memory, None);
            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_buffer_memory, None);
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Copy)]
struct CameraUBO {
    position: Point3<f32>,
    alignment_fix: f32, // see https://vulkan-tutorial.com/en/Uniform_buffers/Descriptor_pool_and_sets#page_Alignment-requirements
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

impl From<&Camera> for CameraUBO {
    fn from(camera: &Camera) -> Self {
        CameraUBO {
            position: camera.position.into(),
            alignment_fix: 0.0,
            view: camera.view,
            proj: camera.projection,
        }
    }
}

// TODO - how to handle layout 140 better?
#[repr(C)]
struct LightUBO {
    position: [f32; 3],
    alignment_fix_1: f32,
    ambient: [f32; 3],
    alignment_fix_2: f32,
    diffuse: [f32; 3],
    alignment_fix_3: f32,
    specular: [f32; 3],
    alignment_fix_4: f32,
}
impl From<Light> for LightUBO {
    fn from(light: Light) -> Self {
        LightUBO {
            position: light.position,
            ambient: light.ambient,
            diffuse: light.diffuse,
            specular: light.specular,
            alignment_fix_1: 0.0,
            alignment_fix_2: 0.0,
            alignment_fix_3: 0.0,
            alignment_fix_4: 0.0,
        }
    }
}

#[repr(C)]
struct LightsUBO {
    count: u32,
    alignment_fix_1: [f32; 3],
    lights: [LightUBO; 2], // hardcoded "2"
}

impl From<&Lights> for LightsUBO {
    fn from(lights: &Lights) -> Self {
        LightsUBO {
            count: lights.lights.len() as u32,
            alignment_fix_1: [0., 0., 0.],
            lights: [
                LightUBO::from(lights.lights[0]),
                LightUBO::from(lights.lights[1]),
            ],
        }
    }
}

struct SyncObjects {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    inflight_fences: Vec<vk::Fence>,
}

// would be good to have graphics_setup here, but as it is mutable this opens up a Pandroras box
pub(crate) struct VulkanGraphicsExecution {
    core: VulkanCore,

    clear_value: [f32; 4],

    uniform_buffers: Vec<UniformBuffer>,
    static_meshes: Vec<VulkanMesh>,
    snow_mesh: Vec<VulkanMesh>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    is_framebuffer_resized: bool,
}

impl VulkanGraphicsExecution {
    pub(crate) fn new(core: VulkanCore, graphics_setup: &VulkanGraphicsSetup) -> Self {
        let uniform_buffers = VulkanGraphicsExecution::create_uniform_buffers(
            &core,
            graphics_setup.swapchain_composite.images.len(),
        );
        let descriptor_sets = VulkanGraphicsExecution::create_descriptor_sets(
            &core.device,
            graphics_setup.descriptor_pool,
            graphics_setup.descriptor_set_layout,
            &uniform_buffers,
            graphics_setup.swapchain_composite.images.len(),
        );
        let sync_objects = VulkanGraphicsExecution::create_sync_objects(&core);

        VulkanGraphicsExecution {
            core,

            clear_value: [0.0, 0.0, 0.0, 0.0],

            uniform_buffers,
            static_meshes: vec![],
            snow_mesh: vec![],
            descriptor_sets,
            command_buffers: vec![],

            image_available_semaphores: sync_objects.image_available_semaphores,
            render_finished_semaphores: sync_objects.render_finished_semaphores,
            in_flight_fences: sync_objects.inflight_fences,
            current_frame: 0,

            is_framebuffer_resized: false,
        }
    }

    fn create_uniform_buffers(
        core: &VulkanCore,
        swapchain_image_count: usize,
    ) -> Vec<UniformBuffer> {
        let mut uniform_buffers = vec![];

        {
            let buffer_size = std::mem::size_of::<CameraUBO>();

            let mut buffers = vec![];
            let mut buffers_memory = vec![];

            for _ in 0..swapchain_image_count {
                let (uniform_buffer, uniform_buffer_memory) = core.create_buffer(
                    buffer_size as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                buffers.push(uniform_buffer);
                buffers_memory.push(uniform_buffer_memory);
            }

            uniform_buffers.push(UniformBuffer {
                buffers,
                buffers_memory,
            });
        }
        {
            let buffer_size = std::mem::size_of::<LightsUBO>();

            let mut buffers = vec![];
            let mut buffers_memory = vec![];

            for _ in 0..swapchain_image_count {
                let (uniform_buffer, uniform_buffer_memory) = core.create_buffer(
                    buffer_size as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                buffers.push(uniform_buffer);
                buffers_memory.push(uniform_buffer_memory);
            }

            uniform_buffers.push(UniformBuffer {
                buffers,
                buffers_memory,
            });
        }
        uniform_buffers
    }

    fn create_descriptor_sets(
        device: &ash::Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        uniforms_buffers: &Vec<UniformBuffer>,
        swapchain_images_size: usize,
    ) -> Vec<vk::DescriptorSet> {
        let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
        for _ in 0..swapchain_images_size {
            layouts.push(descriptor_set_layout);
        }

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: swapchain_images_size as u32,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets!")
        };

        for (i, &descritptor_set) in descriptor_sets.iter().enumerate() {
            let descriptor_buffer_info = [
                vk::DescriptorBufferInfo {
                    buffer: uniforms_buffers[CAMERA_UBO_INDEX].buffers[i],
                    offset: 0,
                    range: std::mem::size_of::<CameraUBO>() as u64,
                },
                vk::DescriptorBufferInfo {
                    buffer: uniforms_buffers[LIGHTS_UBO_INDEX].buffers[i],
                    offset: 0,
                    range: std::mem::size_of::<LightsUBO>() as u64,
                },
            ];

            let descriptor_write_sets = [vk::WriteDescriptorSet {
                dst_set: descritptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: descriptor_buffer_info.len() as u32,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: descriptor_buffer_info.as_ptr(),
                p_texel_buffer_view: ptr::null(),
                ..Default::default()
            }];

            unsafe {
                device.update_descriptor_sets(&descriptor_write_sets, &[]);
            }
        }

        descriptor_sets
    }

    fn create_sync_objects(core: &VulkanCore) -> SyncObjects {
        let mut sync_objects = SyncObjects {
            image_available_semaphores: vec![],
            render_finished_semaphores: vec![],
            inflight_fences: vec![],
        };

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_available_semaphore = core.create_semaphore();
            let render_finished_semaphore = core.create_semaphore();
            let inflight_fence = core.create_fence();

            sync_objects
                .image_available_semaphores
                .push(image_available_semaphore);
            sync_objects
                .render_finished_semaphores
                .push(render_finished_semaphore);
            sync_objects.inflight_fences.push(inflight_fence);
        }

        sync_objects
    }

    pub(crate) fn update_camera(&mut self, camera: &Camera, graphics_setup: &VulkanGraphicsSetup) {
        let ubo: CameraUBO = CameraUBO::from(camera);
        let ubos = [ubo];

        let buffer_size = (std::mem::size_of::<CameraUBO>() * ubos.len()) as u64;

        for current_image in 0..graphics_setup.swapchain_composite.images.len() {
            unsafe {
                let data_ptr =
                    self.core
                        .device
                        .map_memory(
                            self.uniform_buffers[CAMERA_UBO_INDEX].buffers_memory[current_image],
                            0,
                            buffer_size,
                            vk::MemoryMapFlags::empty(),
                        )
                        .expect("Failed to Map Memory") as *mut CameraUBO;

                data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

                self.core.device.unmap_memory(
                    self.uniform_buffers[CAMERA_UBO_INDEX].buffers_memory[current_image],
                );
            }
        }
    }

    pub(crate) fn update_lights(&mut self, lights: &Lights, graphics_setup: &VulkanGraphicsSetup) {
        let ubo: LightsUBO = LightsUBO::from(lights);
        let ubos = [ubo];

        let buffer_size = (std::mem::size_of::<LightsUBO>() * ubos.len()) as u64;

        for current_image in 0..graphics_setup.swapchain_composite.images.len() {
            unsafe {
                let data_ptr =
                    self.core
                        .device
                        .map_memory(
                            self.uniform_buffers[LIGHTS_UBO_INDEX].buffers_memory[current_image],
                            0,
                            buffer_size,
                            vk::MemoryMapFlags::empty(),
                        )
                        .expect("Failed to Map Memory") as *mut LightsUBO;

                data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

                self.core.device.unmap_memory(
                    self.uniform_buffers[LIGHTS_UBO_INDEX].buffers_memory[current_image],
                );
            }
        }
    }

    pub(crate) fn set_clear_value(&mut self, clear_value: [f32; 4]) {
        self.clear_value = clear_value;
    }

    pub(crate) fn set_static_meshes(
        &mut self,
        meshes: &Vec<Mesh>,
        graphics_setup: &VulkanGraphicsSetup,
    ) {
        self.static_meshes = meshes
            .iter()
            .map(|m| self.to_vulkan_mesh(graphics_setup, m))
            .collect();
    }

    pub(crate) fn set_snow_mesh(
        &mut self,
        meshes: &Vec<Mesh>,
        graphics_setup: &VulkanGraphicsSetup,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        self.snow_mesh = meshes
            .iter()
            .map(|m| self.to_vulkan_mesh(graphics_setup, m))
            .collect();

        let last_mesh = self.snow_mesh.last().unwrap();
        (
            last_mesh.instance_buffer.clone(),
            last_mesh.instance_buffer_memory.clone(),
        )
    }

    fn to_vulkan_mesh(&self, graphics_setup: &VulkanGraphicsSetup, mesh: &Mesh) -> VulkanMesh {
        let (vertex_buffer, vertex_buffer_memory) = VulkanGraphicsExecution::create_vertex_buffer(
            &self.core,
            graphics_setup.command_pool,
            &mesh.vertices,
        );
        let (index_buffer, index_buffer_memory) = VulkanGraphicsExecution::create_index_buffer(
            &self.core,
            graphics_setup.command_pool,
            &mesh.indices,
        );
        let indices_no = mesh.indices.len() as u32;
        let (instance_buffer, instance_buffer_memory) =
            VulkanGraphicsExecution::create_vertex_buffer(
                &self.core,
                graphics_setup.command_pool,
                &mesh.instances,
            );
        let instances_no = mesh.instances.len() as u32;
        VulkanMesh {
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            indices_no,
            instance_buffer,
            instance_buffer_memory,
            instances_no,
        }
    }

    pub(crate) fn create_command_buffers(&mut self, graphics_setup: &VulkanGraphicsSetup) {
        let device = &self.core.device;
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            command_buffer_count: graphics_setup.swapchain_composite.framebuffers.len() as u32,
            command_pool: graphics_setup.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        };

        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo {
                p_inheritance_info: ptr::null(),
                flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
                ..Default::default()
            };

            unsafe {
                device
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Failed to begin recording Command Buffer at beginning!");
            }

            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: self.clear_value,
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo {
                render_pass: graphics_setup.render_pass,
                framebuffer: graphics_setup.swapchain_composite.framebuffers[i],
                render_area: vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: graphics_setup.swapchain_composite.extent,
                },
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr(),
                ..Default::default()
            };

            unsafe {
                device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                self.execute_static_objects_pipeline(graphics_setup, i, command_buffer);
                self.execute_snow_pipeline(graphics_setup, i, command_buffer);
                device.cmd_end_render_pass(command_buffer);

                device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to record Command Buffer at Ending!");
            }
        }

        self.command_buffers = command_buffers;
    }

    fn execute_static_objects_pipeline(
        &self,
        graphics_setup: &VulkanGraphicsSetup,
        frame_index: usize,
        command_buffer: vk::CommandBuffer,
    ) {
        let device = &self.core.device;
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_setup.pipeline,
            );

            let descriptor_sets_to_bind = [self.descriptor_sets[frame_index]];
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_setup.pipeline_layout,
                0,
                &descriptor_sets_to_bind,
                &[],
            );

            for mesh in self.static_meshes.iter() {
                let vertex_buffers = [mesh.vertex_buffer, mesh.instance_buffer];
                let offsets = [0_u64, 0_u64];

                device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    mesh.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(
                    command_buffer,
                    mesh.indices_no,
                    mesh.instances_no,
                    0,
                    0,
                    0,
                );
            }
        }
    }

    fn execute_snow_pipeline(
        &self,
        graphics_setup: &VulkanGraphicsSetup,
        frame_index: usize,
        command_buffer: vk::CommandBuffer,
    ) {
        let device = &self.core.device;
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_setup.pipeline,
            );

            let descriptor_sets_to_bind = [self.descriptor_sets[frame_index]];
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_setup.pipeline_layout,
                0,
                &descriptor_sets_to_bind,
                &[],
            );

            for mesh in self.snow_mesh.iter() {
                let vertex_buffers = [mesh.vertex_buffer, mesh.instance_buffer];
                let offsets = [0_u64, 0_u64];

                device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    mesh.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(
                    command_buffer,
                    mesh.indices_no,
                    mesh.instances_no,
                    0,
                    0,
                    0,
                );
            }
        }
    }

    pub(crate) fn draw_frame(
        &mut self,
        graphics_setup: &mut VulkanGraphicsSetup,
        snow_calculated_semaphore: vk::Semaphore,
    ) {
        let device = &self.core.device;
        let wait_fences = [self.in_flight_fences[self.current_frame]];

        unsafe {
            device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence!");
        }

        let (image_index, _is_sub_optimal) = unsafe {
            let result = graphics_setup
                .swapchain_composite
                .loader
                .acquire_next_image(
                    graphics_setup.swapchain_composite.swapchain,
                    std::u64::MAX,
                    self.image_available_semaphores[self.current_frame],
                    vk::Fence::null(),
                );
            match result {
                Ok(image_index) => image_index,
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        self.recreate_swapchain(graphics_setup);
                        return;
                    }
                    _ => panic!("Failed to acquire Swap Chain Image!"),
                },
            }
        };

        let wait_semaphores = [
            self.image_available_semaphores[self.current_frame],
            snow_calculated_semaphore,
        ];
        let wait_stages = [
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::VERTEX_INPUT,
        ];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

        let submit_infos = [vk::SubmitInfo {
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffers[image_index as usize],
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        }];

        unsafe {
            device
                .reset_fences(&wait_fences)
                .expect("Failed to reset Fence!");

            device
                .queue_submit(
                    self.core.graphics_queue,
                    &submit_infos,
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to execute queue submit.");
        }

        let swapchains = [graphics_setup.swapchain_composite.swapchain];

        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &image_index,
            p_results: ptr::null_mut(),
            ..Default::default()
        };

        let result = unsafe {
            graphics_setup
                .swapchain_composite
                .loader
                .queue_present(self.core.present_queue, &present_info)
        };
        let is_resized = match result {
            Ok(_) => self.is_framebuffer_resized,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                _ => panic!("Failed to execute queue present."),
            },
        };
        if is_resized {
            self.is_framebuffer_resized = false;
            self.recreate_swapchain(graphics_setup);
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub(crate) fn cleanup_swapchain(&self, command_pool: vk::CommandPool) {
        unsafe {
            self.core
                .device
                .free_command_buffers(command_pool, &self.command_buffers);
        }
    }

    fn recreate_swapchain(&mut self, graphics_setup: &mut VulkanGraphicsSetup) {
        graphics_setup.recreate_swapchain();
        self.create_command_buffers(graphics_setup);
    }

    pub(crate) fn framebuffer_resized(&mut self) {
        self.is_framebuffer_resized = true;
    }

    fn create_vertex_buffer<T>(
        core: &VulkanCore,
        command_pool: vk::CommandPool,
        data: &[T],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        core.create_data_buffer(
            command_pool,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            data,
        )
    }

    fn create_index_buffer(
        core: &VulkanCore,
        command_pool: vk::CommandPool,
        data: &[u32],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        core.create_data_buffer(command_pool, vk::BufferUsageFlags::INDEX_BUFFER, data)
    }

    pub(crate) fn drop(&mut self) {
        unsafe {
            let device = &self.core.device;
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                device.destroy_semaphore(self.image_available_semaphores[i], None);
                device.destroy_semaphore(self.render_finished_semaphores[i], None);
                device.destroy_fence(self.in_flight_fences[i], None);
            }

            self.static_meshes.iter().for_each(|m| m.drop(&device));
            self.snow_mesh.iter().for_each(|m| m.drop(&device));
            for j in 0..self.uniform_buffers.len() {
                for i in 0..self.uniform_buffers[j].buffers.len() {
                    device.destroy_buffer(self.uniform_buffers[j].buffers[i], None);
                    device.free_memory(self.uniform_buffers[j].buffers_memory[i], None);
                }
            }
        }
    }
}
