use winit::event_loop::EventLoop;

use vulkan::Vulkan;

mod vulkan;

pub mod fs;

fn main() {
    let event_loop = EventLoop::new();

    let app = Vulkan::new(&event_loop);
    app.main_loop(event_loop);
}
