const int iterations = 100;

precision highp float;

varying vec2 texcoord;
varying vec3 v_in;
varying vec3 EP_in;


uniform sampler2D uSampler;
uniform vec2 center;
uniform float zoom;
uniform vec2 c;

uniform int resx;
uniform int resy;
uniform float aspect;
uniform float time;

uniform mat4 rotmatrix;
uniform mat4 posmatrix;

uniform mat4 matrix, normalMatrix;

uniform vec3 pos;
uniform vec3 rot;
uniform float pitch;
uniform float yaw;

const float epsilon = 0.001;
const float fov = 45.0;

float sdSphere( vec3 p, float s )
{
  return length(p)-s;
}

float udBox( vec3 p, vec3 b )
{
  return length(max(abs(p)-b,0.0));
}

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

float hit( vec3 r )
{
  //r = rotate( r, sin(time), cos(time), 0.0 );
  vec3 zn = vec3( r.xyz );
  float rad = 0.0;
  float hit = 0.0;
  float p = 8.0;
  float d = 1.0;
  for( int i = 0; i < 4; i++ )
  {
      rad = length( zn );

      if( rad > 2.0 )
      {
        hit = 0.5 * log(rad) * rad / d;
      }else{

      float th = atan( length( zn.xy ), zn.z );
      float phi = atan( zn.y, zn.x );
      float rado = pow(rad,8.0);
      d = pow(rad, 7.0) * 7.0 * d + 1.0;

      float sint = sin( th * p );
      zn.x = rado * sint * cos( phi * p );
      zn.y = rado * sint * sin( phi * p );
      zn.z = rado * cos( th * p ) ;
      zn += r;
      }

  }

  return hit;

}

vec3 eps = vec3( .1, 0.0, 0.0 );

float distSphere(vec3 p, float r)
{
    return abs(length(p) - r);
}


int pMod(inout float p, float d)
{
    p = mod(p + d, d * 2.0) - d;
    return int( float(p) / float(d) );
}


mat4 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

// p = eye + right*u + up*v;

vec3 eye = vec3(0, 0.5, 1.0);
vec3 up	= vec3(0, 1, 0);
vec3 right = vec3(1, 0, 0);
vec3 forward = normalize(cross(right, up));

vec3 zaxis = normalize(eye - pos);
vec3 xaxis = cross(up, zaxis);
vec3 yaxis = cross(xaxis, -zaxis);

void main(void)
{
//  mat4 rotationX = mat4(1.0), rotationY = mat4(1.0), rotationZ = mat4(1.0);

//  rotationX = rotationMatrix( vec3(1, 0, 0), rot.x);
//  rotationY = rotationMatrix( vec3(0, 1, 0), rot.y);
//  rotationZ = rotationMatrix( vec3(0, 0, 1), rot.z);

//  mat4 rotation = rotationX * rotationY * rotationZ;

  float fov_ratio = (0.5 * aspect) / (tan(radians(fov * 0.5)));
  float u = aspect * (gl_FragCoord.x * 2.0 / resx - 1.0);
  float v = gl_FragCoord.y * 2.0 / resy - 1.0;

  vec3 rayOrigin = eye + pos;
  vec3 rayDirection = (vec4(normalize(forward*fov_ratio + right*u + up*v), 1.0) * normalMatrix).xyz;
  //rayDirection= vec3(rotmatrix * vec4(rayDirection, 1.0));

//  float sinPitch = sin(rot.x);
//  float sinYaw = sin(rot.z);
//  float cosPitch = cos(rot.x);
//  float cosYaw = cos(rot.z);

  //rayOrigin = vec3( sinPitch*sinYaw, cosPitch, -sinPitch*cosYaw ) ;
//  rayDirection =  vec3( -cosPitch*sinYaw + u*cosYaw + v*sinPitch*sinYaw,
//                        sinPitch + v*cosPitch,
//                        cosPitch*cosYaw  + u*sinYaw - v*sinPitch*cosYaw );


  //rayOrigin = EP_in;
  //rayOrigin = vec4(eye, 1.0) * posmatrix;
  //vec3 cameraPos = vec3(vec4(u, v, -1.0, 1.0) * rotmatrix);


  //rayOrigin = vec3( matrix * vec4(rayOrigin, 1.0) );
  //rayDirection = vec3( matrix * vec4(rayDirection, 1.0) );


  //rayDirection = v_in - rayOrigin;

  const int maxSteps = 80;

  float t = 0.0;

  float d = 0.0;
  vec3 r = vec3(0,0,0);

  int a = 0;
  for(int i = 0; i < maxSteps; ++i)
  {
    r = rayOrigin + (rayDirection * t);

    //pMod(r.x, 4096.0);
    pMod(r.z, 2.0);
//    pMod(r.y, 2.0);
    pMod(r.x, 2.0);

    d = hit(r);
    //d = dScene(r);

    if( d < epsilon )
    {
      a = i;
      break;
    }

    t += d;
  }

//  vec3 n = vec3( hit( r + eps ) - hit( r - eps ),
//                 hit( r + eps.yxz ) - hit( r - eps.yxz ),
//                 hit( r + eps.zyx ) - hit( r - eps.zyx ) );

  //gl_FragColor = shade(r,n);
  gl_FragColor = a / (float)maxSteps;
  //gl_FragColor = vec4(r, 1.0);
  //gl_FragColor = vec4(rayDirection, 1.0);

  //gl_FragColor = vec4(normalize(rayDirection) + vec3(0.5, 0.5, 0.5), 1);
}

