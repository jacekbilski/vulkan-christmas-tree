#version 450

struct Light {
    vec3 position;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 projection;
    vec3 position;
} camera;

//layout (set = 0, binding = 1) uniform Lights {
//    Light light[2];
//} lights;

const Light LIGHT1 = Light(vec3(10., -100., 10.), vec3(0.3, 0.3, 0.3), vec3(0.2, 0.2, 0.2), vec3(0., 0., 0.));
const Light LIGHT2 = Light(vec3(5., -6., 2.), vec3(0.2, 0.2, 0.2), vec3(2., 2., 2.), vec3(0.5, 0.5, 0.5));

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragAmbient;
layout(location = 3) in vec3 fragDiffuse;
layout(location = 4) in vec3 fragSpecular;
layout(location = 5) in float fragShininess;

layout(location = 0) out vec4 f_color;

vec3 calcLight(Light light);

void main() {
    vec3 result = vec3(0.0);
//    for (int i = 0; i < 2; i++) {
//        result += calcLight(lights.light[i]);
//    }
    result += calcLight(LIGHT1);
    result += calcLight(LIGHT2);
    f_color = vec4(result, 1.0);
//    f_color = vec4(1.0);
//    f_color = vec4(lights.light[1].diffuse, 1.0);
}

vec3 calcLight(Light light) {
    vec3 ambient = light.ambient * fragAmbient;

    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(light.position - fragPosition);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * light.diffuse * fragDiffuse;

    vec3 viewDir = normalize(camera.position - fragPosition);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), fragShininess);
    vec3 specular = spec * light.specular * fragSpecular;

    return ambient + diffuse + specular;
}
