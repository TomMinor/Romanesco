const int iterations = 100;

precision highp float;

varying vec2 texcoord;

uniform int resx;
uniform int resy;
uniform float aspect;
uniform float time;

uniform mat4 normalMatrix;
uniform vec3 pos;

const float epsilon = 0.001;
const float fov = 75.0;

vec4 shade(vec3 p, vec3 n)
{
  const vec3 lightPos = vec3( .5, .5, -2.0 );

  return 0.2 + max(0.0, dot( normalize(lightPos - p ), n)) * 16.0;
}

vec3 rotate( vec3 pos, float x, float y, float z )
{
  mat3 rotX = mat3( 1.0, 0.0, 0.0, 0.0, cos( x ), -sin( x ), 0.0, sin( x ), cos( x ) );
  mat3 rotY = mat3( cos( y ), 0.0, sin( y ), 0.0, 1.0, 0.0, -sin(y), 0.0, cos(y) );
  mat3 rotZ = mat3( cos( z ), -sin( z ), 0.0, sin( z ), cos( z ), 0.0, 0.0, 0.0, 1.0 );

  return rotX * rotY * rotZ * pos;
}

vec3 bulbPower(in vec3 z, float p)
{
  vec3 zn = z;
  float rad = length(zn);

  // convert to polar coords
  float th = acos(zn.z / rad);
  float phi = atan( zn.y, zn.x );
  //d = pow(rad, p - 1.0) * (p - 1.0) * d + 1.0;

  // scale and rotate the point
  float zr = pow(rad, p);
  th = th * p;
  phi = phi * p;

  // Convert to cartesian
  //float sint = sin(th);
  zn = zr * vec3( sin(th) * cos(phi),
                  sin(th) * sin(phi),
                  cos(th));

  return z;
}

int last = 0;
float escapeLength(in vec3 pos)
{
  const int iterations = 10;
  const float bailout = 2.0;
  const float p = 8.0;

  vec3 z = pos;

  for(int i = 1; i < iterations; i++)
  {
    z = bulbPower(z, p) + pos;

    float r2 = dot(z,z);
    if( (r2 > bailout && last == 0) || (i==last) )
    {
      last = i;
      break;
    }
  }

  return length(z);
}

float gradient = 0.0f;
const float EPS = 0.00001f;
float DE(vec3 p)
{
  vec3 xDir = vec3(1, 0, 0);
  vec3 yDir = vec3(0, 1, 0);
  vec3 zDir = vec3(0, 0, 1);

  const float bailout = 2.0;

  last = 0;
  float r = escapeLength(p);
  if( r*r > bailout) { return 0.0; }

  gradient = (vec3(escapeLength(p + xDir*EPS), escapeLength(p + yDir*EPS), escapeLength(p + zDir*EPS)) -r)/EPS;

  return 0.5 * r * log(r) / length(gradient);
}

float hit( vec3 w )
{
  // mat4 rotate;
  // w = vec4(w, 1.0) * rotate; // How to rotate ~the object~

  const int iterations = 10;
  const float bailout = 2.0;

  vec3 zn = vec3( w.xyz );
  float rad = 0.0;
  float p = 8.0;
  float pd = p - 1.0; // Derivative power
  float d = 1.0;

//  float c = w;
//  float r = length(w);
//  float th = atan(w.y, w.x);
//  float ph = asin(w.z / r);

//  // z orbit distance for AO shading
//  if( r < min_dist ) min_dist = r;

//  // Derivative
//  vec3 dw;
//  float ph_dw = 0.0;
//  float th_dw = 0.0;
///  float r_dw = 1.0;
//  float powR, powRsin;

//  vec3 phase = vec3(sin(time), sin(time), sin(time));

  for( int i = 0; i < iterations; i++ )
  {
      rad = length( zn );
      if( rad > bailout ) {
        break;
      }

      // convert to polar coords
      //float th = acos(zn.z / rad);
      float th = atan( length( zn.xy ), zn.z );
      //float phi = atan( zn.y, zn.x );
      float phi = atan(zn.y, zn.x);
      d = pow(rad, pd) * (pd) * d + 1.0;

      // scale and rotate the point
      float zr = pow(rad, p);
      th = th * p;
      phi = phi * p;

      // Convert to cartesian
      float sint = sin(th);
      zn = zr * vec3( sint * cos(phi),
                      sint * sin(phi),
                      cos(th) );

      zn += w;

      //      // Calculate derivative
      //      powR = p * pow(r, pd);
      //      powRsin = powR * r_dw * sin(ph_dw * pd*ph);
      //      dw = vec3( powRsin * cos(th_dw + pd*th) + 1.0,
      //                 powRsin * sin(th_dw + pd*th),
      //                 powR * r_dw * cos(ph_dw + pd*ph)
      //                );

      //      // Polar coords of derivative dw
      //      r_dw = length(dw);
      //      th_dw = atan(dw.y, dw.x);
      //      ph_dw = acos(dw.z / r_dw);

      //      // Z iteration
      //      powR = pow(r, p);
      //      powRsin = sin(p * ph);
      //      w = vec3( powR * powRsin * cos(p * th),
      //                powR * powRsin * sin(p * th),
      //                powR * cos(p * ph)
      //                      );
      //      w += c;

      //      // The triplex power formula applies the azimuthal angle rotation about the y-axis.
      //      // Constrain this to get some funky effects
      //      //if()

      //      r = length(w);
      //      if( r < min_dist) min_dist = r;
      //      if( r > bailout) break;

      //      th = atan(w.y, w.x) + phase.x;
      //      ph = acos(w.z / r) + phase.y;
  }

  return 0.5 * rad * log(rad) / d;

}

vec3 eps = vec3( .1, 0.0, 0.0 );

int pMod(inout float p, float d)
{
    p = mod(p + d, d * 2.0) - d;
    return int( float(p) / float(d) );
}

// p = eye + right*u + up*v;
vec3 eye = vec3(0, 0.5, -1.0);
vec3 up	= vec3(0, 1, 0);
vec3 right = vec3(1, 0, 0);
vec3 forward = normalize(cross(right, up));

void main(void)
{

  float fov_ratio = (0.5 * aspect) / (tan(radians(fov * 0.5)));
  float u = aspect * (gl_FragCoord.x * 2.0 / resx - 1.0);
  float v = gl_FragCoord.y * 2.0 / resy - 1.0;

  vec3 rayOrigin = eye + pos;
  vec3 rayDirection = (vec4(normalize(forward*fov_ratio + right*u + up*v), 1.0) * normalMatrix).xyz;

  const int maxSteps = 64;

  float t = 0.0;

  float d = 0.0;
  vec3 r = vec3(0,0,0);

  int a = 0;

  int A = 0;
  vec3 colour = vec3(0.25,0.1,1.0);
  float min_dist = 2.0;
  for(int i = 0; i < maxSteps && d < 2.0; ++i)
  {
    r = rayOrigin + (rayDirection * t);

    d = hit(r, min_dist);

    if( d < epsilon)
    {
      a = i;

//      for(int j = 0; j < 8; ++j)
//      {
//        r += rayDirection * d;
//        d = mandelbulb(r);
//        colour += (a / (float)maxSteps) * ( 1.0/8);
//      }

      break;
    }

    t += d;
  }

  //gl_FragColor = vec4(colour, 1.0);
  //gl_FragColor = shade(r,n);
  gl_FragColor = a / (float)maxSteps ;
  //gl_FragColor = vec4(min_dist);
  //gl_FragColor = (a / (float)maxSteps) * 2;
  //gl_FragColor = vec4(r, 1.0);
}

