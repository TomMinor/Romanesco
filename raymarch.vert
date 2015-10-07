attribute vec3 vtxPos;
attribute vec2 vtxUV;

varying vec2 texcoord;

void main(void)
{
  texcoord = vtxUV;
  gl_Position = vec4(vtxPos, 1.0);
}
