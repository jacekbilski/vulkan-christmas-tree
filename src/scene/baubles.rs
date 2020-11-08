use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8, PI};

use cgmath::{vec3, EuclideanSpace, Matrix4, Point3};

use crate::coords::CylindricalPoint3;
use crate::mesh::{Color, InstanceData, Mesh};
use crate::vulkan::{Vertex, VertexIndexType};

const PRECISION: VertexIndexType = 8;
const RADIUS: f32 = 0.2;

struct Bauble {
    center: CylindricalPoint3<f32>,
    color: Color,
}

pub fn create_meshes() -> Vec<Mesh> {
    let (vertices, indices) = gen_sphere();

    let red = Color {
        ambient: [0.1745, 0.01175, 0.01175],
        diffuse: [0.61424, 0.04136, 0.04136],
        specular: [0.727811, 0.626959, 0.626959],
        shininess: 76.8,
    };
    let blue = Color {
        ambient: [0.01175, 0.01175, 0.1745],
        diffuse: [0.04136, 0.04136, 0.61424],
        specular: [0.626959, 0.626959, 0.61424],
        shininess: 76.8,
    };
    let yellow = Color {
        ambient: [0.1745, 0.1745, 0.01175],
        diffuse: [0.61424, 0.61424, 0.04136],
        specular: [0.727811, 0.727811, 0.626959],
        shininess: 76.8,
    };
    let light_blue = Color {
        ambient: [0.01175, 0.1745, 0.1745],
        diffuse: [0.04136, 0.61424, 0.61424],
        specular: [0.626959, 0.727811, 0.727811],
        shininess: 76.8,
    };
    let violet = Color {
        ambient: [0.1745, 0.01175, 0.1745],
        diffuse: [0.61424, 0.04136, 0.61424],
        specular: [0.727811, 0.626959, 0.727811],
        shininess: 76.8,
    };

    let baubles: Vec<Bauble> = vec![
        Bauble {
            center: CylindricalPoint3::new(0., 0., -2.7),
            color: red,
        },
        Bauble {
            center: CylindricalPoint3::new(1.1, -0.5, -1.3),
            color: blue,
        },
        Bauble {
            center: CylindricalPoint3::new(1.1, 1.7, -1.3),
            color: yellow,
        },
        Bauble {
            center: CylindricalPoint3::new(1.5, 1.2, -0.25),
            color: red,
        },
        Bauble {
            center: CylindricalPoint3::new(1.5, -1.7, -0.25),
            color: light_blue,
        },
        Bauble {
            center: CylindricalPoint3::new(2.2, 1.0, 0.85),
            color: light_blue,
        },
        Bauble {
            center: CylindricalPoint3::new(2.2, 3. * FRAC_PI_4, 0.85),
            color: blue,
        },
        Bauble {
            center: CylindricalPoint3::new(2.2, -0.2, 0.85),
            color: red,
        },
        Bauble {
            center: CylindricalPoint3::new(3., FRAC_PI_2, 1.8),
            color: violet,
        },
        Bauble {
            center: CylindricalPoint3::new(3., -FRAC_PI_2, 1.8),
            color: yellow,
        },
        Bauble {
            center: CylindricalPoint3::new(3., -FRAC_PI_4 - 3., 1.8),
            color: red,
        },
        Bauble {
            center: CylindricalPoint3::new(3., 3.6, 1.8),
            color: violet,
        },
        Bauble {
            center: CylindricalPoint3::new(3., 0.2, 1.8),
            color: blue,
        },
        Bauble {
            center: CylindricalPoint3::new(3.6, 1. * FRAC_PI_6, 3.),
            color: light_blue,
        },
        Bauble {
            center: CylindricalPoint3::new(3.6, 2. * FRAC_PI_6, 3.),
            color: red,
        },
        Bauble {
            center: CylindricalPoint3::new(3.6, 4. * FRAC_PI_6, 3.),
            color: blue,
        },
        Bauble {
            center: CylindricalPoint3::new(3.6, 5. * FRAC_PI_6, 3.),
            color: violet,
        },
        Bauble {
            center: CylindricalPoint3::new(3.6, 6. * FRAC_PI_6, 3.),
            color: yellow,
        },
        Bauble {
            center: CylindricalPoint3::new(3.6, 8. * FRAC_PI_6, 3.),
            color: blue,
        },
        Bauble {
            center: CylindricalPoint3::new(3.6, 9. * FRAC_PI_6, 3.),
            color: light_blue,
        },
        Bauble {
            center: CylindricalPoint3::new(3.6, 11. * FRAC_PI_6, 3.),
            color: yellow,
        },
        Bauble {
            center: CylindricalPoint3::new(4., 3. * FRAC_PI_8, 4.1),
            color: light_blue,
        },
        Bauble {
            center: CylindricalPoint3::new(4., 4. * FRAC_PI_8, 4.1),
            color: yellow,
        },
        Bauble {
            center: CylindricalPoint3::new(4., 5. * FRAC_PI_8, 4.1),
            color: blue,
        },
        Bauble {
            center: CylindricalPoint3::new(4., 7. * FRAC_PI_8, 4.1),
            color: violet,
        },
        Bauble {
            center: CylindricalPoint3::new(4., 11. * FRAC_PI_8, 4.1),
            color: red,
        },
        Bauble {
            center: CylindricalPoint3::new(4., 12. * FRAC_PI_8, 4.1),
            color: blue,
        },
        Bauble {
            center: CylindricalPoint3::new(4., 13. * FRAC_PI_8, 4.1),
            color: yellow,
        },
        Bauble {
            center: CylindricalPoint3::new(4., 17. * FRAC_PI_8, 4.1),
            color: red,
        },
        Bauble {
            center: CylindricalPoint3::new(4., 21. * FRAC_PI_8, 4.1),
            color: blue,
        },
    ];

    let instances: Vec<InstanceData> = baubles
        .into_iter()
        .map(|b| {
            let point: Point3<f32> = b.center.into();
            InstanceData {
                color: b.color,
                model: Matrix4::from_translation(point.to_vec()).into(),
                ..Default::default()
            }
        })
        .collect();

    vec![Mesh {
        vertices,
        indices,
        instances,
    }]
}

fn gen_sphere() -> (Vec<Vertex>, Vec<VertexIndexType>) {
    let vertices = gen_vertices();
    let indices = gen_indices();
    (vertices, indices)
}

fn gen_vertices() -> Vec<Vertex> {
    let mut vertices: Vec<Vertex> = Vec::with_capacity(2 * PRECISION.pow(2) as usize);
    let angle_diff = PI / PRECISION as f32;

    vertices.push(Vertex {
        pos: Point3::new(0., RADIUS, 0.).into(),
        norm: vec3(0., 1., 0.).into(),
    });

    for layer in 1..PRECISION {
        let v_angle = angle_diff * layer as f32; // vertically I'm doing only half rotation
        for slice in 0..(2 * PRECISION) {
            let h_angle = angle_diff * slice as f32; // horizontally I'm doing full circle
            let layer_radius = RADIUS * v_angle.sin();
            let vertex = Point3::new(
                layer_radius * h_angle.sin(),
                RADIUS * v_angle.cos(),
                layer_radius * h_angle.cos(),
            );

            vertices.push(Vertex {
                pos: vertex.into(),
                norm: vec3(h_angle.sin(), v_angle.cos(), h_angle.cos()).into(),
            });
        }
    }

    vertices.push(Vertex {
        pos: Point3::new(0., -RADIUS, 0.).into(),
        norm: vec3(0., -1., 0.).into(),
    });

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
        indices.extend(
            [
                find_index(layer - 1, slice),
                find_index(layer, slice),
                find_index(layer, slice + 1),
            ]
            .iter(),
        );
    }

    for layer in 2..PRECISION {
        for slice in 0..2 * PRECISION {
            // midddle layers are actually traapezoids, I need two triangles per slice
            indices.extend(
                [
                    find_index(layer - 1, slice),
                    find_index(layer, slice),
                    find_index(layer, slice + 1),
                ]
                .iter(),
            );
            indices.extend(
                [
                    find_index(layer - 1, slice + 1),
                    find_index(layer - 1, slice),
                    find_index(layer, slice + 1),
                ]
                .iter(),
            );
        }
    }

    layer = PRECISION;
    for slice in 0..2 * PRECISION {
        // last layer has only triangles
        indices.extend(
            [
                find_index(layer - 1, slice + 1),
                find_index(layer - 1, slice),
                find_index(layer, slice),
            ]
            .iter(),
        );
    }

    indices
}
