attribute vec3 vtxPos;
attribute vec2 vtxUV;

in vec2 texcoord;
in vec3 v_in;
in vec3 EP_in;

uniform mat4 rotmatrix;
uniform mat4 posmatrix;

vec3 eye = vec3(0, 0.5, -1.25);

void main(void)
{
  gl_Position = vec4(vtxPos, 1.0);
  texcoord = vtxUV;
  v_in = vec3( rotmatrix * gl_Position );
  EP_in = vec3( posmatrix * vec4(eye,1) );
}
