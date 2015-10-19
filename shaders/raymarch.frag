#version 330

const int maxSteps = 48;
const int iterations = 9;

varying vec2 texcoord;

uniform int resx;
uniform int resy;
uniform float aspect;
uniform float time;

uniform mat4 normalMatrix;
uniform vec3 pos;

const float epsilon = 0.001;
const float fov = 45.0;

float sintime = sin(time);
float costime = cos(time);
float sinTimeSlow = sin(time / 64.0);

vec4 shade(vec3 p, vec3 n)
{
  vec3 lightPos = vec3(sintime, sintime, costime) * vec3( 2.0, 4.0, 2.0 );

  const vec4 colourA = vec4(0.34, 0.078, 1.0, 1.0) * 2.0;
  const vec4 colourB = vec4(0.83, 0.158, 0.192, 1.0) * 2.0;
  float shading = max(0.0, dot( normalize(lightPos - p ), n));

  return mix(colourA, colourB, shading);
}

//vec3 rotate( vec3 pos, float x, float y, float z )
//{
//  mat3 rotX = mat3( 1.0, 0.0, 0.0, 0.0, cos( x ), -sin( x ), 0.0, sin( x ), cos( x ) );
//  mat3 rotY = mat3( cos( y ), 0.0, sin( y ), 0.0, 1.0, 0.0, -sin(y), 0.0, cos(y) );
//  mat3 rotZ = mat3( cos( z ), -sin( z ), 0.0, sin( z ), cos( z ), 0.0, 0.0, 0.0, 1.0 );

//  return rotX * rotY * rotZ * pos;
//}

//vec3 bulbPower(in vec3 z, float p)
//{
//  vec3 zn = z;
//  float rad = length(zn);

//  // convert to polar coords
//  float th = acos(zn.z / rad);
//  float phi = atan( zn.y, zn.x );
//  //d = pow(rad, p - 1.0) * (p - 1.0) * d + 1.0;

//  // scale and rotate the point
//  float zr = pow(rad, p);
//  th = th * p;
//  phi = phi * p;

//  // Convert to cartesian
//  //float sint = sin(th);
//  zn = zr * vec3( sin(th) * cos(phi),
//                  sin(th) * sin(phi),
//                  cos(th));

//  return z;
//}

//int last = 0;
//float escapeLength(in vec3 pos)
//{
//  const float bailout = 2.0;
//  const float p = 8.0;

//  vec3 z = pos;

//  for(int i = 1; i < iterations; i++)
//  {
//    z = bulbPower(z, p) + pos;

//    float r2 = dot(z,z);
//    if( (r2 > bailout && last == 0) || (i==last) )
//    {
//      last = i;
//      break;
//    }
//  }

//  return length(z);
//}

//float gradient = 0.0f;
//const float EPS = 0.00001f;

//float DE(vec3 p)
//{
//  const float bailout = 2.0;

//  last = 0;
//  float r = escapeLength(p);
//  if( r*r > bailout) { return 0.0; }

////  return normalize(vec3(  hit( r + eps ) - hit( r - eps ),
////                          hit( r + eps.yxz ) - hit( r - eps.yxz ),
////                          hit( r + eps.zyx ) - hit( r - eps.zyx ) ) );

////  gradient = (vec3( escapeLength(p + eps),
////                    escapeLength(p + eps.yxz),
////                    escapeLength(p + eps.zyx)) -r) / eps;

//  return 0.5 * r * log(r) / length(gradient);
//}

vec3 eps = vec3( 0.1, 0, 0 );
float hit( vec3 w )
{
  // mat4 rotate;
  // w = vec4(w, 1.0) * rotate; // How to rotate ~the object~
  const float bailout = 2.0;

  vec3 zn = vec3( w.xyz );
  float rad = 0.0;
  float p = (5.0 * abs(sinTimeSlow)) + 3.0;
  float pd = p - 1.0; // Derivative power
  float d = 1.0;

  for( int i = 0; i < iterations; i++ )
  {
      rad = length( zn );
      if( rad > bailout ) {
        break;
      }

      // convert to polar coords
      float th = acos(zn.z / rad) + sinTimeSlow;
      float phi = atan( zn.y, zn.x ) + sinTimeSlow;
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
  }

  return 0.5 * rad * log(rad) / d;
}

//vec3 eps = vec3( 0.1, 0.1, -0.1 );
vec3 getNormal( vec3 r )
{
  return normalize(vec3(  hit( r + eps ) - hit( r - eps ),
                          hit( r + eps.yxz ) - hit( r - eps.yxz ),
                          hit( r + eps.zyx ) - hit( r - eps.zyx ) ) );
}


int pMod(inout float p, float d)
{
    p = mod(p + d, d * 2.0) - d;
    return int( float(p) / float(d) );
}

float sdSphere( vec3 p, float s )
{
  return length(p)-s;
}

//vec3 refract(vec3 r)
//{
//  //http://www.math.ubc.ca/~cass/courses/m309-03a/text/refraction/refraction.html
//}

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

  float t = 0.0;

  float d = 0.0;
  vec3 r = vec3(0,0,0);

  int a = 0;
  vec3 colour = vec3(0.25,0.1,1.0);

  vec3 arse = rayOrigin + (rayDirection * 800);

  for(int i = 0; i < maxSteps && d < 2.0; ++i)
  {
    r = rayOrigin + (rayDirection * t);

    d = hit(r);

    if( d < epsilon )
    {
      a = i;
      break;
    }

    t += d;
  }

  // Check if we missed
  if( t > 5 )
  {
    gl_FragColor = vec4(mix( vec3(0.72, 0.91, 0.92),
                             vec3(0.9, 0.9, 0.9),
                             v), 1.0);
  }
  else
  {
    vec3 nrm = getNormal(r);

    float iterValue = maxSteps;
    iterValue = a / iterValue;
    //gl_FragColor = vec4(colour, 1.0);

    gl_FragColor = shade(r, nrm) * (0.25 + vec4(iterValue));

    //gl_FragColor = vec4(iterValue, iterValue, iterValue, 1.0);
    //gl_FragColor = vec4(nrm, 1.0);
  }

//  a = 0;
//  rayOrigin = r + (rayDirection * 0.5);
//  t = 0;
//  for(int i = 0; i < maxSteps && d < 2.0; ++i)
//  {
//    r = rayOrigin + (rayDirection * t);

//    d = hit(r);

//    if( d < epsilon )
//    {
//      a = i;
//      break;
//    }

//    t += d;
//  }


  //gl_FragColor = (a / (float)maxSteps);
//  gl_FragColor = vec4(arse, 1.0);
}

