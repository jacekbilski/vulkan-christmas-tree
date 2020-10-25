use cgmath::{Matrix4, Point3, SquareMatrix};

use crate::coords::SphericalPoint3;
use crate::mesh::{Color, InstanceData, Mesh};
use crate::scene::camera::Camera;
use crate::vulkan::{Vertex, Vulkan};

pub mod camera;

const CLEAR_VALUE: [f32; 4] = [0.015_7, 0., 0.360_7, 1.0];

const VERTICES_DATA: [Vertex; 4] = [
    Vertex {
        pos: [-0.5, 0.0, -0.5],
    },
    Vertex {
        pos: [0.5, 0.0, -0.5],
    },
    Vertex {
        pos: [0.5, 0.0, 0.5],
    },
    Vertex {
        pos: [-0.5, 0.0, 0.5],
    },
];
const INDICES_DATA: [u32; 6] = [0, 2, 1, 3, 2, 0];

pub struct Scene {
    pub camera: Camera,
}

impl Scene {
    pub fn setup(vulkan: &mut Vulkan, window: &winit::window::Window) -> Self {
        vulkan.set_clear_value(CLEAR_VALUE);

        let camera_position: SphericalPoint3<f32> =
            SphericalPoint3::from(Point3::new(1.1, 1.1, 1.1));
        let look_at = Point3::new(0.0, -0.1, 0.0);
        let camera = Camera::new(
            camera_position,
            look_at,
            [window.inner_size().width, window.inner_size().height],
        );
        vulkan.update_camera(&camera);

        let color = Color {
            color: [1.0, 1.0, 1.0],
        };
        let mesh = Mesh {
            vertices: Vec::from(VERTICES_DATA),
            indices: Vec::from(INDICES_DATA),
            instances: vec![InstanceData {
                color,
                model: Matrix4::identity(),
            }],
        };
        vulkan.set_meshes(&vec![mesh]);

        Self { camera }
    }

    pub fn rotate_camera_horizontally(&mut self, angle: f32, vulkan: &mut Vulkan) {
        self.camera.rotate_horizontally(angle, vulkan);
    }
}
