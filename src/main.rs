use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::memory::pool::{PotentialDedicatedAllocation, StdMemoryPoolAlloc};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError, ColorSpace, FullscreenExclusive, PresentMode, Surface, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};
use vulkano_win::VkSurfaceBuild;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

// settings
pub const SCR_WIDTH: u32 = 1920;
pub const SCR_HEIGHT: u32 = 1080;

mod vs {
    vulkano_shaders::shader! {ty: "vertex", path: "src/shaders/shader.vert"}
}

mod fs {
    vulkano_shaders::shader! {ty: "fragment", path: "src/shaders/shader.frag"}
}

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

vulkano::impl_vertex!(Vertex, position);

fn main() {
    // bummer, I cannot store PhysicalDevice directly, there's a problem with lifetime
    let (instance, physical_device_index, device, queue) = init_vulkan();
    let event_loop = EventLoop::new();
    let surface = setup_window(&instance, &event_loop);
    let (mut swapchain, images) = setup_swapchain(&instance, physical_device_index, &device, &queue, &surface);
    let render_pass = setup_render_pass(&device, &mut swapchain);
    let pipeline = create_pipeline(&device, &render_pass);
    let mut dynamic_state = create_dynamic_state();
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    let vertex_buffer = create_vertex_buffer(&device);
    let clear_values = vec![[0.015_7, 0., 0.360_7, 1.0].into()];

    let mut recreating_swapchain_necessary = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    event_loop.run(move |event, _, control_flow| {
        match event {
            winit::event::Event::WindowEvent { event: winit::event::WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            winit::event::Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                recreating_swapchain_necessary = true;
            }
            winit::event::Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();
                if recreating_swapchain_necessary {
                    // I cannot assign directly to variables, see https://github.com/rust-lang/rfcs/issues/372
                    let (new_swapchain, new_framebuffers) = recreate_swapchain(surface.clone(), swapchain.clone(), render_pass.clone(), &mut dynamic_state);
                    swapchain = new_swapchain;
                    framebuffers = new_framebuffers;
                    recreating_swapchain_necessary = false;
                }
                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreating_swapchain_necessary = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };
                if suboptimal {
                    recreating_swapchain_necessary = true;
                }

                let mut command_builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
                command_builder
                    .begin_render_pass(framebuffers[image_num].clone(), false, clear_values.clone())
                    .unwrap()

                    .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ())
                    .unwrap()

                    .end_render_pass()
                    .unwrap();
                let command_buffer = command_builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreating_swapchain_necessary = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => ()
        }
    });
}

fn init_vulkan() -> (Arc<Instance>, usize, Arc<Device>, Arc<Queue>) {
    let extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &extensions, None)
        .expect("failed to create instance");
    let physical_device_index = PhysicalDevice::enumerate(&instance).position(|_device| true).expect("no device available");
    let physical = PhysicalDevice::from_index(&instance, physical_device_index).expect("no device available");
    println!("Got physical device, name: {}, type: {:?}", physical.name(), physical.ty());
    let queue_family = physical.queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");
    let (device, mut queues) = {
        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        Device::new(physical, &Features::none(), &device_ext, [(queue_family, 0.5)].iter().cloned())
            .expect("failed to create device")
    };
    let queue = queues.next().unwrap();
    (instance, physical_device_index, device, queue)
}

fn setup_window(instance: &Arc<Instance>, event_loop: &EventLoop<()>) -> Arc<Surface<Window>> {
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();
    surface.window().set_inner_size(PhysicalSize::new(SCR_WIDTH, SCR_HEIGHT));
    surface.window().set_title("Vulkan Christmas Tree");
    surface
}

fn setup_swapchain(instance: &Arc<Instance>, physical_device_index: usize, device: &Arc<Device>, queue: &Arc<Queue>, surface: &Arc<Surface<Window>>) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
    let caps = surface.capabilities(PhysicalDevice::from_index(&instance, physical_device_index).unwrap())
        .expect("failed to get surface capabilities");
    let dimensions: [u32; 2] = surface.window().inner_size().into();
    let alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let format = caps.supported_formats[0].0;
    Swapchain::new(device.clone(), surface.clone(),
                   caps.min_image_count, format, dimensions, 1, ImageUsage::color_attachment(), queue,
                   SurfaceTransform::Identity, alpha, PresentMode::Fifo, FullscreenExclusive::Default,
                   true, ColorSpace::SrgbNonLinear)
        .expect("failed to create swapchain")
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

fn create_pipeline(device: &Arc<Device>, render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>) -> Arc<GraphicsPipeline<SingleBufferDefinition<Vertex>, Box<dyn PipelineLayoutAbstract + Send + Sync>, Arc<dyn RenderPassAbstract + Send + Sync>>> {
    let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
    let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");
    Arc::new(GraphicsPipeline::start()
        // Defines what kind of vertex input is expected.
        .vertex_input_single_buffer::<Vertex>()
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

fn create_vertex_buffer(device: &Arc<Device>) -> Arc<CpuAccessibleBuffer<[Vertex], PotentialDedicatedAllocation<StdMemoryPoolAlloc>>> {
    let vertex1 = Vertex { position: [-0.5, -0.5] };
    let vertex2 = Vertex { position: [0.0, 0.5] };
    let vertex3 = Vertex { position: [0.5, -0.25] };
    CpuAccessibleBuffer::from_iter(
        device.clone(), BufferUsage::all(), false, vec![vertex1, vertex2, vertex3].into_iter())
        .unwrap()
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
    mut dynamic_state: &mut DynamicState) -> (Arc<Swapchain<Window>>, Vec<Arc<dyn FramebufferAbstract + Send + Sync>>) {
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
    let framebuffers = window_size_dependent_setup(
        &new_images,
        render_pass.clone(),
        &mut dynamic_state,
    );
    (new_swapchain, framebuffers)
}
