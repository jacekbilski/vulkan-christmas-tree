#version 450

layout(binding = 0) uniform Camera {
    mat4 model;
    mat4 view;
    mat4 projection;
} camera;

// per-vertex data
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

// per-instance data
layout(location = 2) in vec3 colour;

layout(location = 0) out vec3 fragColour;

void main() {
    vec4 pos = camera.model * vec4(position, 1.0);
    gl_Position = camera.projection * camera.view * pos;
    fragColour = colour;
}
