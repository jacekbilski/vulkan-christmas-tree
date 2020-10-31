use std::f32::consts::PI;
use std::ops::Neg;

use cgmath::{vec3, EuclideanSpace, Euler, Matrix4, Point3, Rad, Vector3};
use rand::distributions::Uniform;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::mesh::{Color, InstanceData, Mesh};
use crate::vulkan::Vertex;

const SNOW_X_MIN: f32 = -10.;
const SNOW_X_MAX: f32 = 10.;
const SNOW_Y_MIN: f32 = -10.;
const SNOW_Y_MAX: f32 = 5.;
const SNOW_Z_MIN: f32 = -10.;
const SNOW_Z_MAX: f32 = 10.;

const MAX_SNOWFLAKES: usize = 5_000;

struct Snowflake {
    position: Point3<f32>,
    rotation: Vector3<Rad<f32>>,
}

pub fn create_meshes() -> Vec<Mesh> {
    let color = Color {
        ambient: [1.0, 1.0, 1.0],
        diffuse: [0.623960, 0.686685, 0.693872],
        specular: [0.5, 0.5, 0.5],
        shininess: 225.0,
    };
    let snowflakes = gen_snowflakes();
    let (vertices, indices) = gen_snowflake_mesh();
    let instances = gen_instances(&snowflakes, color);
    vec![Mesh {
        vertices,
        indices,
        instances,
    }]
}

fn gen_snowflake_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let radius: f32 = 0.05;
    let normal: Vector3<f32> = vec3(1., 0., 0.);
    let mut vertices: Vec<Vertex> = vec![];

    let angle_diff = PI / 3 as f32;

    for i in 0..6 {
        let angle = i as f32 * angle_diff;
        // upper side
        vertices.push(Vertex {
            pos: Point3::new(0., radius * angle.cos(), radius * angle.sin()).into(),
            norm: normal.into(),
        });
        // bottom side
        vertices.push(Vertex {
            pos: Point3::new(-0., -radius * angle.cos(), -radius * angle.sin()).into(),
            norm: normal.neg().into(),
        });
    }
    let indices: Vec<u32> = vec![
        8, 4, 0, 10, 6, 2, // upper side
        1, 5, 9, 3, 7, 11, // bottom side
    ];

    (vertices, indices)
}

fn gen_snowflakes() -> Vec<Snowflake> {
    let mut snowflakes: Vec<Snowflake> = Vec::with_capacity(MAX_SNOWFLAKES as usize);
    let x_range = Uniform::new(SNOW_X_MIN, SNOW_X_MAX);
    let y_range = Uniform::new(SNOW_Y_MIN, SNOW_Y_MAX);
    let z_range = Uniform::new(SNOW_Z_MIN, SNOW_Z_MAX);
    let angle_range = Uniform::new(0., 2. * PI);
    let mut rng = SmallRng::from_entropy();
    for _i in 0..MAX_SNOWFLAKES {
        let x_position = rng.sample(x_range);
        let y_position = rng.sample(y_range);
        let z_position = rng.sample(z_range);
        let x_rotation = Rad(rng.sample(angle_range));
        let y_rotation = Rad(rng.sample(angle_range));
        let z_rotation = Rad(rng.sample(angle_range));
        let position = Point3::new(x_position, y_position, z_position);
        let rotation = vec3(x_rotation, y_rotation, z_rotation);
        snowflakes.push(Snowflake { position, rotation });
    }
    snowflakes
}

fn gen_instances(snowflakes: &Vec<Snowflake>, color: Color) -> Vec<InstanceData> {
    let mut instances: Vec<InstanceData> = Vec::with_capacity(snowflakes.len());
    for snowflake in snowflakes {
        let rotation = Matrix4::from(Euler {
            x: snowflake.rotation.x,
            y: snowflake.rotation.y,
            z: snowflake.rotation.z,
        });
        let translation = Matrix4::from_translation(snowflake.position.to_vec());
        let model = translation * rotation;
        instances.push(InstanceData { model, color });
    }
    instances
}
