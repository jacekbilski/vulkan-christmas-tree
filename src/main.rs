use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::instance::InstanceExtensions;

fn main() {
    let instance = Instance::new(None, &InstanceExtensions::none(), None)
        .expect("failed to create instance");
    println!("Got instance: {:?}", instance);

    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");
    println!("Got physical device: {:?}", physical);
}
