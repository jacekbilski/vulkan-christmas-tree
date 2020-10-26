#version 450
#extension GL_ARB_separate_shader_objects : enable

struct Light {
    vec3 position;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

layout(set = 0, binding = 1) uniform LightsUBO {
    int count;
    Light light[2];
} lights;

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragAmbient;
layout(location = 3) in vec3 fragDiffuse;
layout(location = 4) in vec3 fragSpecular;
layout(location = 5) in float fragShininess;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragDiffuse, 1.0);
}
