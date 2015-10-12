attribute vec3 vtxPos;
attribute vec2 vtxUV;

varying vec2 texcoord;
varying vec3 v_in;
varying vec3 EP_in;

uniform mat4 matrix;

vec3 eye = vec3(0, 0.5, -1.25);

void main(void)
{
  gl_Position = vec4(vtxPos, 1.0);
  texcoord = vtxUV;
  v_in = vec3( matrix * gl_Position );
  EP_in = vec3( matrix * vec4(eye,1) );
}
