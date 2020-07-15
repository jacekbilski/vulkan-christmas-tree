use std::sync::Arc;
use std::time::Instant;

use image::{ImageBuffer, Rgba};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::image::{Dimensions, StorageImage};
use vulkano::image::ImageUsage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::memory::pool::{PotentialDedicatedAllocation, StdMemoryPoolAlloc};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain};
use vulkano::sync::GpuFuture;
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

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
    let mut start = Instant::now();
    let extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &extensions, None)
        .expect("failed to create instance");
    println!("Got instance: {:?}", instance);
    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");
    println!("Got physical device: {:?}", physical);
    for family in physical.queue_families() {
        println!("Found a queue family with {:?} queue(s)", family.queues_count());
    }
    let queue_family = physical.queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");
    println!("Got queue family: {:?}", queue_family);
    let (device, mut queues) = {
        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        Device::new(physical, &Features::none(), &device_ext, [(queue_family, 0.5)].iter().cloned())
            .expect("failed to create device")
    };
    println!("Got a device: {:?}", device);
    let queue = queues.next().unwrap();
    println!("Got a single queue: {:?}", queue);
    let mut duration = start.elapsed().as_millis();
    println!("Vulkan initialized in {} ms", duration);

    start = Instant::now();
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();
    duration = start.elapsed().as_millis();
    println!("Window created in {} ms", duration);

    start = Instant::now();
    let caps = surface.capabilities(physical)
        .expect("failed to get surface capabilities");
    let dimensions = caps.current_extent.unwrap_or([SCR_WIDTH, SCR_HEIGHT]);
    let alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let format = caps.supported_formats[0].0;
    let (swapchain, images) =
        Swapchain::new(device.clone(), surface.clone(),
                       caps.min_image_count, format, dimensions, 1, ImageUsage::color_attachment(), &queue,
                       SurfaceTransform::Identity, alpha, PresentMode::Fifo, FullscreenExclusive::Default,
                       true, ColorSpace::SrgbNonLinear)
            .expect("failed to create swapchain");
    duration = start.elapsed().as_millis();
    println!("Swapchain created in {} ms", duration);

    start = Instant::now();
    let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: Format::R8G8B8A8Unorm,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    ).unwrap());
    let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
    let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");
    let pipeline = Arc::new(GraphicsPipeline::start()
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
        .unwrap());
    duration = start.elapsed().as_millis();
    println!("Pipeline created in {} ms", duration);

    let vertex_buffer = create_vertex_buffer(&device);

    let dynamic_state = DynamicState {
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [1024.0, 1024.0],
            depth_range: 0.0..1.0,
        }]),
        ..DynamicState::none()
    };

    let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
                                  Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
    println!("Created a Vulkan StorageImage: {:?}", image);

    let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
        .add(image.clone()).unwrap()
        .build().unwrap());

    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, (0..1024 * 1024 * 4).map(|_| 0u8))
        .expect("failed to create buffer");

    start = Instant::now();
    let mut command_builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
    command_builder
        .begin_render_pass(framebuffer.clone(), false, vec![[0.015_7, 0., 0.360_7, 1.0].into()])
        .unwrap()

        .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ())
        .unwrap()

        .end_render_pass()
        .unwrap()

        .copy_image_to_buffer(image.clone(), buf.clone())
        .unwrap();
    let command_buffer = command_builder.build().unwrap();
    duration = start.elapsed().as_millis();
    println!("Command buffer created in {} ms", duration);

    start = Instant::now();
    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();
    duration = start.elapsed().as_millis();
    println!("Command buffer executed in {} ms", duration);

    start = Instant::now();
    let buffer_content = buf.read().unwrap();
    let result = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    result.save("image.png").unwrap();
    duration = start.elapsed().as_millis();
    println!("Result copied and saved in {} ms", duration);

    event_loop.run(|event, _, control_flow| {
        match event {
            winit::event::Event::WindowEvent { event: winit::event::WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            _ => ()
        }
    });
}

fn create_vertex_buffer(device: &Arc<Device>) -> Arc<CpuAccessibleBuffer<[Vertex], PotentialDedicatedAllocation<StdMemoryPoolAlloc>>> {
    let vertex1 = Vertex { position: [-0.5, -0.5] };
    let vertex2 = Vertex { position: [0.0, 0.5] };
    let vertex3 = Vertex { position: [0.5, -0.25] };
    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(), BufferUsage::all(), false, vec![vertex1, vertex2, vertex3].into_iter())
        .unwrap();
    vertex_buffer
}
