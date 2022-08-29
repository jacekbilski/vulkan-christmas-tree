use cgmath::Point3;
use winit::dpi::PhysicalSize;

use crate::color_mesh::ColorMesh;
use crate::coords::SphericalPoint3;
use crate::scene::camera::Camera;
use crate::scene::lights::Lights;
use crate::textured_mesh::TexturedMesh;
use crate::vulkan::Vulkan;

mod baubles;
pub mod camera;
mod ground;
pub mod lights;
pub mod snow;
mod tree;

const CLEAR_VALUE: [f32; 4] = [0.015_7, 0., 0.360_7, 1.0];

pub struct Scene {
    pub camera: Camera,
}

impl Scene {
    pub fn setup(vulkan: &mut Vulkan, window: &winit::window::Window) -> Self {
        vulkan.set_clear_value(CLEAR_VALUE);
        let camera = Scene::setup_camera(vulkan, window);
        Scene::setup_lights(vulkan);
        Scene::setup_meshes(vulkan);

        Self { camera }
    }

    fn setup_camera(vulkan: &mut Vulkan, window: &winit::window::Window) -> Camera {
        let camera_position: SphericalPoint3<f32> = SphericalPoint3::new(18., 1.7, 0.9);
        let look_at: Point3<f32> = Point3::new(0., 1., 0.);
        let camera = Camera::new(camera_position, look_at, window.inner_size());
        vulkan.update_camera(&camera);
        camera
    }

    fn setup_lights(vulkan: &mut Vulkan) {
        let mut lights = Lights::setup();
        lights.add(
            Point3::new(10., -100., 10.),
            [0.3, 0.3, 0.3],
            [0.2, 0.2, 0.2],
            [0., 0., 0.],
        );
        lights.add(
            Point3::new(5., -6., 2.),
            [0.2, 0.2, 0.2],
            [2., 2., 2.],
            [0.5, 0.5, 0.5],
        );
        vulkan.update_lights(&lights);
    }

    fn setup_meshes(vulkan: &mut Vulkan) {
        let mut color_meshes: Vec<ColorMesh> = Vec::new();
        color_meshes.extend(baubles::create_meshes());
        color_meshes.extend(tree::create_meshes());
        let mut textured_meshes: Vec<TexturedMesh> = Vec::new();
        textured_meshes.extend(ground::create_meshes());
        vulkan.set_static_meshes(&color_meshes, &textured_meshes);
        let (snowflakes, snow_meshes) = snow::create_meshes();
        vulkan.set_snow_mesh(&snowflakes, &snow_meshes);
        vulkan.scene_complete();
    }

    pub fn rotate_camera_horizontally(&mut self, angle: f32, vulkan: &mut Vulkan) {
        self.camera.rotate_horizontally(angle, vulkan);
    }

    pub fn rotate_camera_vertically(&mut self, angle: f32, vulkan: &mut Vulkan) {
        self.camera.rotate_vertically(angle, vulkan);
    }

    pub fn change_camera_distance(&mut self, distance: f32, vulkan: &mut Vulkan) {
        self.camera.change_distance(distance, vulkan);
    }

    pub(crate) fn framebuffer_resized(&mut self, new_size: PhysicalSize<u32>, vulkan: &mut Vulkan) {
        self.camera.framebuffer_resized(new_size, vulkan);
    }
}
