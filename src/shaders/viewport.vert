#version 430 core

layout(location=0) in vec2 vtxPos;
layout(location=1) in vec2 vtxUV;

layout(location=0) out vec2 texcoord;

void main(void)
{
	texcoord = vtxUV;
	gl_Position = vec4(vtxPos, 0.0, 1.0);
}
