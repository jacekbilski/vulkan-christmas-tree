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

layout(location = 0) in vec3 fragColour;
layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(fragColour, 1.0);
}
