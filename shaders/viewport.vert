#version 150

in vec3 vtxPos;
in vec2 vtxUV;

out vec2 texcoord;

void main(void)
{
	texcoord = vtxUV;
	gl_Position = vec4(vtxPos, 1.0);
}
