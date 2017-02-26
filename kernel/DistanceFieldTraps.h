#ifndef DISTANCEFIELDTRAPS_H__
#define DISTANCEFIELDTRAPS_H__

#include "DistanceFieldMaths.h"

//// Geometric orbit trap. Creates the 'cube' look.
//float trap(vec3 p){
//	return  length(p.x-0.5-0.5*sin(time/10.0)); // <- cube forms
//	//return  length(p.x-1.0);
//	//return length(p.xz-vec2(1.0,1.0))-0.05; // <- tube forms
//	//return length(p); // <- no trap
//}

class orbitTrap
{
public:
    __device__ orbitTrap() {;}

    __device__ inline virtual void trap( float3 _p ) = 0;

    __device__ inline virtual float getTrapValue() = 0;

private:

};

class CrossTrap
{
public:
    __device__ inline CrossTrap(float _size = 0.05f)
        : m_dist(_size)
    {;}

    __device__ inline void trap( float3 _p )
    {
            if( fabs( _p.x ) < m_dist) {  m_dist = fabs( _p.x ); }
       else if( fabs( _p.y ) < m_dist) {  m_dist = fabs( _p.y ); }
       else if( fabs( _p.z ) < m_dist) {  m_dist = fabs( _p.z ); }
    }

    __device__ inline float getTrapValue()
    {
        return sqrt(m_dist);
    }

private:
    float m_dist;
};

class SphereTrap
{
public:
    __device__ inline SphereTrap(float _size = 0.5f)
        : m_dist(1e20), m_size(_size)
    {;}

    __device__ inline void trap( float3 _p )
    {
        float3 tmp = make_float3(_p.x - m_size, _p.y, _p.z);
        m_dist = dot(tmp,tmp);
    }

    __device__ inline float getTrapValue()
    {
        return sqrt(m_dist);
    }

private:
    float m_dist, m_size;
};

#endif
