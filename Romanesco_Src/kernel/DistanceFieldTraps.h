#ifndef DISTANCEFIELDTRAPS_H__
#define DISTANCEFIELDTRAPS_H__

#include "DistanceFieldMaths.h"

class OrbitTrap
{
public:
    __device__ OrbitTrap() {;}

    __device__ virtual void trap( float3 _p ) = 0;

    __device__ virtual float getTrapValue() = 0;

private:

};

class CrossTrap
{
public:
    __device__ CrossTrap(float _size = 0.05f)
        : m_dist(_size)
    {;}

    __device__ void trap( float3 _p )
    {
            if( fabs( _p.x ) < m_dist) {  m_dist = fabs( _p.x ); }
       else if( fabs( _p.y ) < m_dist) {  m_dist = fabs( _p.y ); }
       else if( fabs( _p.z ) < m_dist) {  m_dist = fabs( _p.z ); }
    }

    __device__ float getTrapValue()
    {
        return sqrt(m_dist);
    }

private:
    float m_dist;
};

class SphereTrap
{
public:
    __device__ SphereTrap(float _size = 0.5f)
        : m_dist(1e20), m_size(_size)
    {;}

    __device__ void trap( float3 _p )
    {
        float3 tmp = make_float3(_p.x - m_size, _p.y, _p.z);
        m_dist = dot(tmp,tmp);
    }

    __device__ float getTrapValue()
    {
        return sqrt(m_dist);
    }

private:
    float m_dist, m_size;
};

#endif
