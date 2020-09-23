use std::f32::consts::PI;
use std::sync::Arc;

use cgmath::{Point3, vec3};
use vulkano::buffer::{BufferUsage, ImmutableBuffer};
use vulkano::device::Queue;
use vulkano::sync::GpuFuture;

use crate::mesh::{InstanceData, Mesh, Vertex, VertexIndexType};

pub fn create_mesh(queue: Arc<Queue>) -> Mesh {
    let precision = 8 as u32;
    let radius = 0.2 as f32;

    let mut vertices: Vec<Vertex> = Vec::with_capacity(2 * precision.pow(2) as usize);
    let mut indices: Vec<VertexIndexType> = Vec::with_capacity(3 * 4 * precision.pow(2) as usize);

    gen_sphere(&mut vertices, &mut indices, Point3::new(0., 0., 0.), radius, precision);

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

fn gen_sphere(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, center: Point3<f32>, radius: f32, precision: u32) {
    gen_vertices(vertices, center, radius, precision);
    gen_indices(indices, precision)
}

fn gen_vertices(vertices: &mut Vec<Vertex>, center: Point3<f32>, radius: f32, precision: u32) {
    let angle_diff = PI / precision as f32;

    vertices.push(Vertex { position: Point3::new(center.x, center.y + radius, center.z).into(), normal: vec3(0., 1., 0.).into() });

    for layer in 1..precision {
        let v_angle = angle_diff * layer as f32;   // vertically I'm doing only half rotation
        for slice in 0..(2 * precision) {
            let h_angle = angle_diff * slice as f32;   // horizontally I'm doing full circle
            let layer_radius = radius * v_angle.sin();
            let vertex = Point3::new(center.x + layer_radius * h_angle.sin(), center.y + radius * v_angle.cos(), center.z + layer_radius * h_angle.cos());

            vertices.push(Vertex { position: vertex.into(), normal: vec3(h_angle.sin(), v_angle.cos(), h_angle.cos()).into() });
        }
    }

    vertices.push(Vertex { position: Point3::new(center.x, center.y - radius, center.z).into(), normal: vec3(0., -1., 0.).into() });
}

fn gen_indices(indices: &mut Vec<u32>, precision: u32) {
    let find_index = |layer: u32, slice: u32| {
        // layers [0] and [precision] have only 1 vertex
        if layer == 0 {
            0
        } else if layer == precision {
            (layer - 1) * 2 * precision + 1
        } else {
            (layer - 1) * 2 * precision + 1 + slice % (2 * precision)
        }
    };

    // I'm generating indices for triangles drawn between this and previous layers of vertices
    let mut layer = 1;
    for slice in 0..2 * precision {
        // first layer has only triangles
        indices.extend([find_index(layer - 1, slice), find_index(layer, slice), find_index(layer, slice + 1)].iter());
    }

    for layer in 2..precision {
        for slice in 0..2 * precision {
            // midddle layers are actually traapezoids, I need two triangles per slice
            indices.extend([find_index(layer - 1, slice), find_index(layer, slice), find_index(layer, slice + 1)].iter());
            indices.extend([find_index(layer - 1, slice + 1), find_index(layer - 1, slice), find_index(layer, slice + 1)].iter());
        }
    }

    layer = precision;
    for slice in 0..2 * precision {
        // last layer has only triangles
        indices.extend([find_index(layer - 1, slice + 1), find_index(layer - 1, slice), find_index(layer, slice)].iter());
    }
}
