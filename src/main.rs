use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::{ClearValue, Format};
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

    let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
                                  Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
    println!("Created a Vulkan StorageImage: {:?}", image);

    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, (0 .. 1024 * 1024 * 4).map(|_| 0u8))
        .expect("failed to create buffer");

    // clearing the image with "blue" colour
    let mut command_builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
    command_builder
        .clear_color_image(image.clone(), ClearValue::Float([0.0, 0.0, 1.0, 1.0])).unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone()).unwrap();
    let command_buffer = command_builder.build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();
}
