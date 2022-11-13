#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform CameraUBO {
    vec3 position;
    mat4 view;
    mat4 projection;
} camera;

// per-vertex data
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

// per-instance data
layout (location = 3) in mat4 model;

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;

void main() {
    vec4 pos = model * vec4(position, 1.0);
    gl_Position = camera.projection * camera.view * pos;
    fragPosition = vec3(gl_Position);
    fragNormal = normalize(mat3(transpose(inverse(model))) * normal);
    fragTexCoord = texCoord;
}
