#version 430 core

layout(location=0) in vec2 texcoord;

layout(location=0) out vec4 colour;

uniform sampler2D viewportBuffer;

void main(void)
{
    colour = texture2D(viewportBuffer, texcoord);
}
