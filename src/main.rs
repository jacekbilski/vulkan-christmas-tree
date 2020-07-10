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
}
