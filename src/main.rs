use std::collections::HashSet;
use std::sync::Arc;

use cgmath::{Deg, Matrix4, perspective, Point3, vec3};
use vulkano::buffer::{BufferUsage, CpuBufferPool};
use vulkano::buffer::cpu_pool::CpuBufferPoolSubbuffer;
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, PersistentDescriptorSetBuf};
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::instance::debug::{DebugCallback, MessageSeverity, MessageType};
use vulkano::memory::pool::StdMemoryPool;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::pipeline::vertex::OneVertexOneInstanceDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError, ColorSpace, FullscreenExclusive, PresentMode, Surface, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture, SharingMode};
use vulkano_win::VkSurfaceBuild;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::desktop::EventLoopExtDesktop;
use winit::window::{Window, WindowBuilder};

use crate::coords::SphericalPoint3;
use crate::mesh::{InstanceData, Vertex};
use crate::scene::Scene;

mod coords;
mod mesh;
mod scene;

// settings
pub const SCR_WIDTH: u32 = 1920;
pub const SCR_HEIGHT: u32 = 1080;

const VALIDATION_LAYERS: &[&str] = &[
    "VK_LAYER_KHRONOS_validation"
];

fn required_device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    }
}

struct QueueFamilyIndices {
    graphics_family: i32,
    present_family: i32,
    compute_family: i32,
}
impl QueueFamilyIndices {
    fn new() -> Self {
        Self { graphics_family: -1, present_family: -1, compute_family: -1 }
    }

    fn is_complete(&self) -> bool {
        self.graphics_family >= 0 && self.present_family >= 0 && self.compute_family >= 0
    }
}

mod vs {
    vulkano_shaders::shader! {ty: "vertex", path: "src/shaders/shader.vert"}
}

mod fs {
    vulkano_shaders::shader! {ty: "fragment", path: "src/shaders/shader.frag"}
}

#[derive(Copy, Clone)]
#[allow(unused)]
pub struct Camera {
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

struct App {
    // instance: Arc<Instance>,
    #[allow(unused)]
    debug_callback: DebugCallback,

    surface: Arc<Surface<Window>>,

    // bummer, I cannot store PhysicalDevice directly, there's a problem with lifetime
    // physical_device_index: usize,
    device: Arc<Device>,

    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,
    // compute_queue: Arc<Queue>,

    swapchain: Arc<Swapchain<Window>>,
    // swapchain_images: Vec<Arc<SwapchainImage<Window>>>,

    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    graphics_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    dynamic_state: DynamicState,

    framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,

    scene: Scene,

    // this should be of type Arc<dyn DescriptorSetsCollection + Send + Sync>
    uniform_buffers: Arc<PersistentDescriptorSet<((), PersistentDescriptorSetBuf<CpuBufferPoolSubbuffer<Camera, Arc<StdMemoryPool>>>)>>,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreating_swapchain_necessary: bool,
}

impl App {
    pub fn initialize() -> (Self, EventLoop<()>) {
        let instance = Self::create_instance();

        let debug_callback = Self::setup_debug_callback(&instance);
        let event_loop = EventLoop::new();
        let surface = Self::setup_window(&instance, &event_loop);
        let physical_device_index = Self::select_physical_device(&instance, &surface);
        let (device, graphics_queue, present_queue, _compute_queue) = Self::setup_device(&instance, &surface, physical_device_index);
        let (mut swapchain, swapchain_images) = Self::setup_swapchain(&instance, physical_device_index, &device, &graphics_queue, &present_queue, &surface);
        let render_pass = Self::setup_render_pass(&device, &mut swapchain);
        let pipeline = Self::create_pipeline(&device, &render_pass);
        let mut dynamic_state = Self::create_dynamic_state();
        let framebuffers = Self::window_size_dependent_setup(&swapchain_images, render_pass.clone(), &mut dynamic_state);
        let scene = Scene::setup(graphics_queue.clone());
        let uniform_buffers = Self::create_camera_ubo(&device, pipeline.clone(), swapchain.dimensions());

        let recreating_swapchain_necessary = false;
        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        (Self {
            // instance,
            debug_callback,

            surface,

            // physical_device_index,
            device,

            graphics_queue,
            present_queue,
            // compute_queue,

            swapchain,
            // swapchain_images,

            render_pass,
            graphics_pipeline: pipeline,

            dynamic_state,

            framebuffers,

            scene,

            uniform_buffers,

            previous_frame_end,
            recreating_swapchain_necessary,
        }, event_loop)
    }

    fn create_instance() -> Arc<Instance> {
        let mut required_extensions = vulkano_win::required_extensions();
        required_extensions.ext_debug_utils = true;
        Instance::new(None, &required_extensions, VALIDATION_LAYERS.iter().cloned())
            .expect("failed to create instance")
    }

    fn setup_debug_callback(instance: &Arc<Instance>) -> DebugCallback {
        let severity = MessageSeverity {
            error: true,
            warning: true,
            information: true,
            verbose: true,
        };
        let types = MessageType {
            general: false,
            performance: true,
            validation: true,
        };
        DebugCallback::new(&instance, severity, types, |msg| {
            println!("validation layer: {:?}", msg.description);
        }).expect("Failed to register DebugCallback")
    }

    fn setup_window(instance: &Arc<Instance>, event_loop: &EventLoop<()>) -> Arc<Surface<Window>> {
        WindowBuilder::new()
            .with_title("Vulkan Christmas Tree")
            .with_inner_size(PhysicalSize::new(SCR_WIDTH, SCR_HEIGHT))
            .build_vk_surface(&event_loop, instance.clone())
            .expect("Failed to create window surface")
    }

    fn select_physical_device(instance: &Arc<Instance>, surface: &Arc<Surface<Window>>) -> usize {
        PhysicalDevice::enumerate(&instance)
            .position(|device| Self::is_device_suitable(surface, &device))
            .expect("no suitable devices available")
    }

    fn is_device_suitable(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> bool {
        let indices = Self::find_queue_families(surface, device);
        let extensions_supported = Self::does_device_supports_required_extensions(device);

        let swap_chain_adequate = if extensions_supported {
            let capabilities = surface.capabilities(*device)
                .expect("failed to get surface capabilities");
            !capabilities.supported_formats.is_empty() &&
                capabilities.present_modes.iter().next().is_some()
        } else {
            false
        };

        indices.is_complete() && extensions_supported && swap_chain_adequate
    }

    fn find_queue_families(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices::new();
        for (i, queue_family) in device.queue_families().enumerate() {
            if queue_family.supports_graphics() {
                indices.graphics_family = i as i32;
            }

            if surface.is_supported(queue_family).unwrap() {
                indices.present_family = i as i32;
            }

            if queue_family.supports_compute() {
                indices.compute_family = i as i32;
            }

            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    fn does_device_supports_required_extensions(device: &PhysicalDevice) -> bool {
        let available_extensions = DeviceExtensions::supported_by_device(*device);
        let device_extensions = required_device_extensions();
        available_extensions.intersection(&device_extensions) == device_extensions
    }

    fn setup_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>, Arc<Queue>) {
        let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).expect("no device available");
        println!("Got physical device, name: {}, type: {:?}", physical_device.name(), physical_device.ty());
        let queue_family_indices = Self::find_queue_families(&surface, &physical_device);

        let unique_families: HashSet<i32> = [queue_family_indices.graphics_family, queue_family_indices.present_family, queue_family_indices.compute_family].iter().cloned().collect();
        let queue_families = unique_families
            .iter()
            .map(|i| (physical_device.queue_families().nth(*i as usize).unwrap(), 1.0));

        let (device, mut queues) = Device::new(
            physical_device,
            &Features::none(),
            &required_device_extensions(),
            queue_families)
                .expect("failed to create device");
        let graphics_queue = queues.next().unwrap();
        let present_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());
        let compute_queue = queues.next().unwrap_or_else(|| present_queue.clone());
        (device, graphics_queue, present_queue, compute_queue)
    }

    fn setup_swapchain(
        instance: &Arc<Instance>,
        physical_device_index: usize,
        device: &Arc<Device>,
        graphics_queue: &Arc<Queue>,
        present_queue: &Arc<Queue>,
        surface: &Arc<Surface<Window>>
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let capabilities = surface.capabilities(PhysicalDevice::from_index(&instance, physical_device_index).unwrap())
            .expect("failed to get surface capabilities");
        let format = capabilities.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        let sharing: SharingMode = if graphics_queue.family().id() != present_queue.family().id() {
            vec![graphics_queue, present_queue].as_slice().into()
        } else {
            graphics_queue.into()
        };
        let alpha = capabilities.supported_composite_alpha.iter().next().unwrap();
        Swapchain::new(
            device.clone(),
            surface.clone(),
            capabilities.min_image_count,
            format,
            dimensions,
            1,
            ImageUsage::color_attachment(),
            sharing,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear
        ).expect("failed to create swapchain")
    }

    fn setup_render_pass(device: &Arc<Device>, swapchain: &Arc<Swapchain<Window>>) -> Arc<dyn RenderPassAbstract + Send + Sync> {
        Arc::new(vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    ).unwrap())
    }

    fn create_pipeline(device: &Arc<Device>, render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
        let vs = vs::Shader::load(device.clone()).expect("failed to create vertex shader module");
        let fs = fs::Shader::load(device.clone()).expect("failed to create fragment shader module");
        Arc::new(GraphicsPipeline::start()
            // Defines what kind of vertex input is expected.
            .vertex_input(OneVertexOneInstanceDefinition::<Vertex, InstanceData>::new())
            // The vertex shader.
            .vertex_shader(vs.main_entry_point(), ())
            // Defines the viewport (explanations below).
            .viewports_dynamic_scissors_irrelevant(1)
            // The fragment shader.
            .fragment_shader(fs.main_entry_point(), ())
            // This graphics pipeline object concerns the first pass of the render pass.
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            // Now that everything is specified, we call `build`.
            .build(device.clone())
            .unwrap())
    }

    fn create_dynamic_state() -> DynamicState {
        DynamicState {
            line_width: None,
            viewports: None,
            scissors: None,
            compare_mask: None,
            write_mask: None,
            reference: None,
        }
    }

    fn create_camera_ubo(
        device: &Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        window_size: [u32; 2],
    ) -> Arc<PersistentDescriptorSet<((), PersistentDescriptorSetBuf<CpuBufferPoolSubbuffer<Camera, Arc<StdMemoryPool>>>)>> {
        let buffer_pool = CpuBufferPool::<Camera>::new(device.clone(), BufferUsage::uniform_buffer());
        let position: SphericalPoint3<f32> = SphericalPoint3::new(18., 1.7, 0.9);
        let look_at: Point3<f32> = Point3::new(0., 1., 0.);

        let camera = Camera {
            view: Matrix4::look_at(position.into(), look_at, vec3(0.0, 1.0, 0.0)),
            projection: perspective(Deg(45.0), window_size[0] as f32 / window_size[1] as f32, 0.1, 100.0),
        };

        let buffer = buffer_pool.next(camera).unwrap();

        let layout = pipeline.descriptor_set_layout(0).unwrap();
        let set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(buffer)
                .unwrap()
                .build()
                .unwrap()
        );
        set
    }

    fn window_size_dependent_setup(
        images: &[Arc<SwapchainImage<Window>>],
        render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
        dynamic_state: &mut DynamicState,
    ) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
        let dimensions = images[0].dimensions();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };
        dynamic_state.viewports = Some(vec![viewport]);

        images
            .iter()
            .map(|image| {
                Arc::new(
                    Framebuffer::start(render_pass.clone())
                        .add(image.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                ) as Arc<dyn FramebufferAbstract + Send + Sync>
            })
            .collect::<Vec<_>>()
    }

    fn recreate_swapchain(
        surface: Arc<Surface<Window>>,
        swapchain: Arc<Swapchain<Window>>,
        render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
        mut dynamic_state: &mut DynamicState,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<dyn FramebufferAbstract + Send + Sync>>) {
        // Get the new dimensions of the window.
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        let (new_swapchain, new_images) =
            match swapchain.recreate_with_dimensions(dimensions) {
                Ok(r) => r,
                // This error tends to happen when the user is manually resizing the window.
                // Simply restarting the loop is the easiest way to fix this issue.
                Err(SwapchainCreationError::UnsupportedDimensions) => panic!("Unsupported dimensions: {:?}", dimensions),
                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
            };

        // Because framebuffers contains an Arc on the old swapchain, we need to
        // recreate framebuffers as well.
        let framebuffers = Self::window_size_dependent_setup(
            &new_images,
            render_pass.clone(),
            &mut dynamic_state,
        );
        (new_swapchain, framebuffers)
    }

    pub fn run(&mut self, mut event_loop: EventLoop<()>) {
        event_loop.run_return(|event, _, control_flow| {
            match event {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                    *control_flow = ControlFlow::Exit;
                }
                Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                    self.recreating_swapchain_necessary = true;
                }
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput {
                        input: KeyboardInput {
                            virtual_keycode: Some(virtual_code),
                            state: ElementState::Pressed,
                            ..
                        },
                        ..
                    },
                    ..
                } => {
                    match virtual_code {
                        VirtualKeyCode::Escape => {
                            *control_flow = ControlFlow::Exit;
                        }
                        _ => ()
                    }
                }
                Event::RedrawEventsCleared => {
                    self.draw_frame()
                }
                _ => ()
            }
        });
    }

    fn draw_frame(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();
        if self.recreating_swapchain_necessary {
            // I cannot assign directly to existing variables, see https://github.com/rust-lang/rfcs/issues/372
            let (new_swapchain, new_framebuffers) = Self::recreate_swapchain(self.surface.clone(), self.swapchain.clone(), self.render_pass.clone(), &mut self.dynamic_state);
            self.swapchain = new_swapchain;
            self.framebuffers = new_framebuffers;
            self.recreating_swapchain_necessary = false;
        }
        let (image_num, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreating_swapchain_necessary = true;
                    return;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };
        if suboptimal {
            self.recreating_swapchain_necessary = true;
        }

        let command_buffer = self.scene.draw(
            self.device.clone(),
            self.graphics_queue.clone(),
            self.framebuffers[image_num].clone(),
            self.graphics_pipeline.clone(),
            &self.dynamic_state,
            self.uniform_buffers.clone(),
        );

        let future = self.previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.graphics_queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.present_queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                self.recreating_swapchain_necessary = true;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }
}

fn main() {
    let (mut app, event_loop) = App::initialize();
    app.run(event_loop);
}
