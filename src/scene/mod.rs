use cgmath::Point3;

use crate::coords::SphericalPoint3;
use crate::scene::camera::Camera;
use crate::vulkan::Vulkan;

pub mod camera;

pub struct Scene {
    pub camera: Camera,
}

impl Scene {
    pub fn setup(vulkan: &mut Vulkan, window: &winit::window::Window) -> Self {
        let camera_position: SphericalPoint3<f32> =
            SphericalPoint3::from(Point3::new(1.1, 1.1, 1.1));
        let look_at = Point3::new(0.0, -0.1, 0.0);
        let camera = Camera::new(
            camera_position,
            look_at,
            [window.inner_size().width, window.inner_size().height],
        );
        vulkan.update_camera(&camera);
        Self { camera }
    }

    pub fn rotate_camera_horizontally(&mut self, angle: f32, vulkan: &mut Vulkan) {
        self.camera.rotate_horizontally(angle, vulkan);
    }
}
