#version 150

//in vec2 texcoord;
uniform sampler2D viewportBuffer;

out vec4 colour;

void main(void)
{
    //colour = texture2D(viewportBuffer, texcoord);
	colour = vec4(1.0,1.0,0.0,1.0);
}
