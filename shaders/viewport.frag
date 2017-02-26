#version 330

in vec2 texcoord;
uniform sampler2D viewportBuffer;

layout(location = 0) out vec4 colour;

void main(void)
{
    colour = texture2D(viewportBuffer, texcoord);
}
