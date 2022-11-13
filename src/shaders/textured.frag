#version 450
#extension GL_ARB_separate_shader_objects : enable

struct Light {
    vec3 position;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

layout(set = 0, binding = 0) uniform CameraUBO {
    vec3 position;
    mat4 view;
    mat4 projection;
} camera;

layout(set = 0, binding = 1) uniform LightsUBO {
    int count;
    Light light[2];
} lights;

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

vec3 calcLight(Light light);

void main() {
    vec3 result = vec3(0.0);
    for (int i = 0; i < lights.count; i++) {
        result += calcLight(lights.light[i]);
    }
    outColor = vec4(result, 1.0);
}

vec3 calcLight(Light light) {
    vec3 ambient = light.ambient;

    vec3 lightDir = normalize(light.position - fragPosition);
    float diff = max(dot(fragNormal, lightDir), 0.0);
    vec3 diffuse = diff * light.diffuse;

    vec3 viewDir = normalize(camera.position - fragPosition);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = max(dot(fragNormal, halfwayDir), 0.0);
    vec3 specular = spec * light.specular;

    return ambient + diffuse + specular;
}
