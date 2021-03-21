use cgmath::{perspective, vec3, Deg, Matrix4, Point3};
use winit::dpi::PhysicalSize;

use crate::coords::SphericalPoint3;
use crate::vulkan::Vulkan;

pub struct Camera {
    pub view: Matrix4<f32>,
    pub projection: Matrix4<f32>,
    pub position: SphericalPoint3<f32>,
    look_at: Point3<f32>,
}

impl Camera {
    pub fn new(
        position: SphericalPoint3<f32>,
        look_at: Point3<f32>,
        window_size: PhysicalSize<u32>,
    ) -> Self {
        Camera {
            view: Camera::view(position, look_at),
            projection: Camera::set_projection(window_size),
            position,
            look_at,
        }
    }

    fn set_projection(window_size: PhysicalSize<u32>) -> Matrix4<f32> {
        perspective(
            Deg(45.0),
            window_size.width as f32 / window_size.height as f32,
            0.1,
            100.0,
        )
    }

    fn view(position: SphericalPoint3<f32>, look_at: Point3<f32>) -> Matrix4<f32> {
        Matrix4::look_at_rh(position.into(), look_at, vec3(0.0, 1.0, 0.0))
    }

    pub fn rotate_horizontally(&mut self, angle: f32, vulkan: &mut Vulkan) {
        self.position.phi += angle;
        self.view = Camera::view(self.position, self.look_at);
        vulkan.update_camera(&self);
    }

    pub fn rotate_vertically(&mut self, angle: f32, vulkan: &mut Vulkan) {
        self.position.theta += angle;
        self.view = Camera::view(self.position, self.look_at);
        vulkan.update_camera(&self);
    }

    pub(crate) fn framebuffer_resized(&mut self, new_size: PhysicalSize<u32>, vulkan: &mut Vulkan) {
        self.projection = Camera::set_projection(new_size);
        vulkan.update_camera(&self);
    }
}
