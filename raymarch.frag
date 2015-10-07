const int iterations = 100;

precision highp float;

varying vec2 texcoord;

uniform sampler2D uSampler;
uniform vec2 center;
uniform float zoom;
uniform vec2 c;

uniform int resx;
uniform int resy;
uniform float aspect;
uniform float time;

void main(void)
{
  // Mandelbrot
  vec2 C = vec2(aspect, 1.0) * (texcoord - vec2(0.5, 0.5)) * vec2( sin(time), sin(time) ) - center;
  vec2 z = C;

  gl_FragColor = vec4(0,0,0,1);

  for(float i = 0.0; i < float(iterations); i += 1.0)
  {
    z = vec2( z.x*z.x - z.y*z.y, 2.0 * z.x * z.y) + C;
    if(dot(z,z) > 4.0)
    {
//      gl_FragColor = texture2D(uSampler, vec2(i / float(iterations), 0.5));
      gl_FragColor = i / float(iterations);
      break;
    }
  }

  //gl_FragColor = vec4(texcoord, 0.0, 1.0);
}
