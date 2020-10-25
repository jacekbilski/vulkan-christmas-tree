use cgmath::Point3;

use crate::coords::SphericalPoint3;
use crate::mesh::Mesh;
use crate::scene::camera::Camera;
use crate::vulkan::Vulkan;

mod baubles;
pub mod camera;
mod ground;

const CLEAR_VALUE: [f32; 4] = [0.015_7, 0., 0.360_7, 1.0];

pub struct Scene {
    pub camera: Camera,
}

impl Scene {
    pub fn setup(vulkan: &mut Vulkan, window: &winit::window::Window) -> Self {
        vulkan.set_clear_value(CLEAR_VALUE);
        let camera = Scene::setup_camera(vulkan, window);
        Scene::setup_meshes(vulkan);

        Self { camera }
    }

    fn setup_camera(vulkan: &mut Vulkan, window: &winit::window::Window) -> Camera {
        let camera_position: SphericalPoint3<f32> = SphericalPoint3::new(18., 1.7, 0.9);
        let look_at: Point3<f32> = Point3::new(0., 1., 0.);
        let camera = Camera::new(
            camera_position,
            look_at,
            [window.inner_size().width, window.inner_size().height],
        );
        vulkan.update_camera(&camera);
        camera
    }

    fn setup_meshes(vulkan: &mut Vulkan) {
        let mut meshes: Vec<Mesh> = Vec::new();
        meshes.push(ground::create_mesh());
        meshes.push(baubles::create_mesh());
        vulkan.set_meshes(&meshes);
    }

    pub fn rotate_camera_horizontally(&mut self, angle: f32, vulkan: &mut Vulkan) {
        self.camera.rotate_horizontally(angle, vulkan);
    }
}
