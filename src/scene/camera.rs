use cgmath::{perspective, vec3, Deg, Matrix4, Point3};

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
        window_size: [u32; 2],
    ) -> Self {
        Camera {
            view: Camera::view(position, look_at),
            projection: perspective(
                Deg(45.0),
                window_size[0] as f32 / window_size[1] as f32,
                0.1,
                100.0,
            ),
            position,
            look_at,
        }
    }

    fn view(position: SphericalPoint3<f32>, look_at: Point3<f32>) -> Matrix4<f32> {
        Matrix4::look_at(position.into(), look_at, vec3(0.0, 1.0, 0.0))
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
}
