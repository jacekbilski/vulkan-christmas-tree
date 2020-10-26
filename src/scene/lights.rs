use cgmath::Point3;

#[derive(Debug, Copy, Clone)]
pub struct Light {
    pub position: [f32; 3],
    pub ambient: [f32; 3],
    pub diffuse: [f32; 3],
    pub specular: [f32; 3],
}

pub struct Lights {
    pub lights: Vec<Light>,
}

impl Lights {
    pub fn setup() -> Self {
        Lights { lights: vec![] }
    }

    pub fn add(
        &mut self,
        position: Point3<f32>,
        ambient: [f32; 3],
        diffuse: [f32; 3],
        specular: [f32; 3],
    ) {
        let light = Light {
            position: position.into(),
            ambient,
            diffuse,
            specular,
        };
        self.lights.push(light);
    }
}
