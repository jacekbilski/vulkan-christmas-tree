#version 450
#extension GL_ARB_separate_shader_objects : enable

struct Light {
    vec3 position;

    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
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

layout(binding = 2) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

vec4 calcLight(Light light);

void main() {
    vec4 result = vec4(0.0);
    for (int i = 0; i < lights.count; i++) {
        result += calcLight(lights.light[i]);
    }
    outColor = result;
}

vec4 calcLight(Light light) {
    vec4 ambient = light.ambient * texture(texSampler, fragTexCoord);

    vec3 lightDir = normalize(light.position - fragPosition);
    //    float diff = max(dot(fragNormal, lightDir), 0.0);
    //    vec4 diffuse = diff * light.diffuse * texture(texSampler, fragTexCoord);

    vec3 viewDir = normalize(camera.position - fragPosition);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = max(dot(fragNormal, halfwayDir), 0.0);
    vec4 specular = spec * light.specular * texture(texSampler, fragTexCoord);

    return 2.5 * ambient + specular;
}
