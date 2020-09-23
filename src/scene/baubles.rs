use std::f32::consts::PI;
use std::sync::Arc;

use cgmath::{Point3, vec3};
use vulkano::buffer::{BufferUsage, ImmutableBuffer};
use vulkano::device::Queue;
use vulkano::sync::GpuFuture;

use crate::mesh::{InstanceData, Mesh, Vertex, VertexIndexType};

const PRECISION: VertexIndexType = 8;
const RADIUS: f32 = 0.2;

pub fn create_mesh(queue: Arc<Queue>) -> Mesh {
    let (vertices, indices) = gen_sphere();

    let (vertex_buffer, vertex_future) = ImmutableBuffer::from_iter(
        vertices.into_iter(), BufferUsage::vertex_buffer(), queue.clone())
        .unwrap();

    let (index_buffer, index_future) = ImmutableBuffer::from_iter(
        indices.iter().cloned(), BufferUsage::index_buffer(), queue.clone()).unwrap();

    let instances: Vec<InstanceData> = vec![
        InstanceData { colour: [0.61424, 0.04136, 0.04136] }
    ];

    let (instances_buffer, instances_future) = ImmutableBuffer::from_iter(
        instances.into_iter(), BufferUsage::vertex_buffer(), queue.clone())
        .unwrap();

    vertex_future.flush().unwrap();
    index_future.flush().unwrap();
    instances_future.flush().unwrap();
    Mesh { vertex_buffer, index_buffer, instances_buffer }
}

fn gen_sphere() -> (Vec<Vertex>, Vec<VertexIndexType>) {
    let vertices = gen_vertices();
    let indices = gen_indices();
    (vertices, indices)
}

fn gen_vertices() -> Vec<Vertex> {
    let mut vertices: Vec<Vertex> = Vec::with_capacity(2 * PRECISION.pow(2) as usize);
    let angle_diff = PI / PRECISION as f32;

    vertices.push(Vertex { position: Point3::new(0., RADIUS, 0.).into(), normal: vec3(0., 1., 0.).into() });

    for layer in 1..PRECISION {
        let v_angle = angle_diff * layer as f32;   // vertically I'm doing only half rotation
        for slice in 0..(2 * PRECISION) {
            let h_angle = angle_diff * slice as f32;   // horizontally I'm doing full circle
            let layer_radius = RADIUS * v_angle.sin();
            let vertex = Point3::new(layer_radius * h_angle.sin(), RADIUS * v_angle.cos(), layer_radius * h_angle.cos());

            vertices.push(Vertex { position: vertex.into(), normal: vec3(h_angle.sin(), v_angle.cos(), h_angle.cos()).into() });
        }
    }

    vertices.push(Vertex { position: Point3::new(0., -RADIUS, 0.).into(), normal: vec3(0., -1., 0.).into() });

    vertices
}

fn gen_indices() -> Vec<VertexIndexType> {
    let mut indices: Vec<VertexIndexType> = Vec::with_capacity(3 * 4 * PRECISION.pow(2) as usize);
    let find_index = |layer: VertexIndexType, slice: VertexIndexType| {
        // layers [0] and [PRECISION] have only 1 vertex
        if layer == 0 {
            0
        } else if layer == PRECISION {
            (layer - 1) * 2 * PRECISION + 1
        } else {
            (layer - 1) * 2 * PRECISION + 1 + slice % (2 * PRECISION)
        }
    };

    // I'm generating indices for triangles drawn between this and previous layers of vertices
    let mut layer = 1;
    for slice in 0..2 * PRECISION {
        // first layer has only triangles
        indices.extend([find_index(layer - 1, slice), find_index(layer, slice), find_index(layer, slice + 1)].iter());
    }

    for layer in 2..PRECISION {
        for slice in 0..2 * PRECISION {
            // midddle layers are actually traapezoids, I need two triangles per slice
            indices.extend([find_index(layer - 1, slice), find_index(layer, slice), find_index(layer, slice + 1)].iter());
            indices.extend([find_index(layer - 1, slice + 1), find_index(layer - 1, slice), find_index(layer, slice + 1)].iter());
        }
    }

    layer = PRECISION;
    for slice in 0..2 * PRECISION {
        // last layer has only triangles
        indices.extend([find_index(layer - 1, slice + 1), find_index(layer - 1, slice), find_index(layer, slice)].iter());
    }

    indices
}
