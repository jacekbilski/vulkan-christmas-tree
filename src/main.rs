use std::sync::Arc;

use image::{ImageBuffer, Rgba};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::Format;
use vulkano::framebuffer::Framebuffer;
use vulkano::image::{Dimensions, StorageImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::instance::InstanceExtensions;
use vulkano::sync::GpuFuture;

fn main() {
    let instance = Instance::new(None, &InstanceExtensions::none(), None)
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
        Device::new(physical, &Features::none(), &DeviceExtensions::none(),
                    [(queue_family, 0.5)].iter().cloned()).expect("failed to create device")
    };
    println!("Got a device: {:?}", device);

    let queue = queues.next().unwrap();
    println!("Got a single queue: {:?}", queue);

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

    let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
                                  Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
    println!("Created a Vulkan StorageImage: {:?}", image);

    let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
        .add(image.clone()).unwrap()
        .build().unwrap());

    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, (0 .. 1024 * 1024 * 4).map(|_| 0u8))
        .expect("failed to create buffer");

    let mut command_builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
    command_builder
        .begin_render_pass(framebuffer.clone(), false, vec![[0.015_7, 0., 0.360_7, 1.0].into()])
        .unwrap()
        .end_render_pass()
        .unwrap();
    let command_buffer = command_builder.build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();
}
