use cgmath::{perspective, vec3, Deg, Matrix4, Point3};

use crate::coords::SphericalPoint3;

pub struct Camera {
    pub view: Matrix4<f32>,
    pub projection: Matrix4<f32>,
    pub position: SphericalPoint3<f32>,
}

impl Camera {
    pub fn new(
        position: SphericalPoint3<f32>,
        look_at: Point3<f32>,
        window_size: [u32; 2],
    ) -> Self {
        Camera {
            view: Matrix4::look_at(position.into(), look_at, vec3(0.0, -1.0, 0.0)),
            projection: perspective(
                Deg(45.0),
                window_size[0] as f32 / window_size[1] as f32,
                0.1,
                100.0,
            ),
            position,
        }
    }
}
