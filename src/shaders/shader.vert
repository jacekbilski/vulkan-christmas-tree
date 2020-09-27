#version 450

layout(binding = 0) uniform Camera {
    mat4 view;
    mat4 projection;
    vec3 position;
} camera;

// per-vertex data
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

// per-instance data
layout(location = 2) in mat4 model;
layout(location = 6) in vec3 ambient;
layout(location = 7) in vec3 diffuse;
layout(location = 8) in vec3 specular;
layout(location = 9) in float shininess;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragAmbient;
layout(location = 3) out vec3 fragDiffuse;
layout(location = 4) out vec3 fragSpecular;
layout(location = 5) out float fragShininess;

void main() {
    vec4 pos = model * vec4(position, 1.0);
    gl_Position = camera.projection * camera.view * pos;
    fragPosition = vec3(gl_Position);
    fragAmbient = ambient;
    fragDiffuse = diffuse;
    fragSpecular = specular;
    fragShininess = shininess;
    fragNormal = mat3(transpose(inverse(model))) * normal;
}
