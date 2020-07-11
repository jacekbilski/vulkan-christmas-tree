use std::sync::Arc;

use image::{ImageBuffer, Rgba};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::{ClearValue, Format};
use vulkano::image::{Dimensions, StorageImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::instance::InstanceExtensions;
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;

mod cs {
    vulkano_shaders::shader!{ty: "compute", path: "src/shaders/mandelbrot.glsl"}
}

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

    let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
                                  Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
    println!("Created a Vulkan StorageImage: {:?}", image);

    let shader = cs::Shader::load(device.clone())
        .expect("failed to create shader module");

    let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
        .expect("failed to create compute pipeline"));

    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
    let descriptor_set = Arc::new(PersistentDescriptorSet::start(layout.clone())
        .add_image(image.clone()).unwrap()
        .build().unwrap()
    );

    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, (0 .. 1024 * 1024 * 4).map(|_| 0u8))
        .expect("failed to create buffer");

    let mut command_builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
    command_builder
        .dispatch([1024 / 8, 1024 / 8, 1], compute_pipeline.clone(), descriptor_set.clone(), ()).unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone()).unwrap();
    let command_buffer = command_builder.build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();
}
