use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::Format;
use vulkano::image::{Dimensions, StorageImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::instance::InstanceExtensions;

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
}
