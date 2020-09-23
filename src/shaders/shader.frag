#version 450

struct Light {
    vec3 position;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

layout (binding = 1) uniform Lights {
    Light light[2];
} lights;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragAmbient;
layout(location = 2) in vec3 fragDiffuse;
layout(location = 3) in vec3 fragSpecular;
layout(location = 4) in float fragShininess;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(fragDiffuse, 1.0);
}
