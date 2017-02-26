#include "Mandelbulb_SDFOP.h"

static const std::vector<Argument> args = {
};

Mandelbulb_SDFOP::Mandelbulb_SDFOP() :
    BaseSDFOP::BaseSDFOP()
{
m_returnType = ReturnType::Float;
}

Mandelbulb_SDFOP::~Mandelbulb_SDFOP()
{

}

std::string Mandelbulb_SDFOP::getFunctionName()
{
    return "mandelbulb";
}

std::string Mandelbulb_SDFOP::getSource()
{
    std::string mandelbulb_hit_src =
               R"(
                   float3 zn  = vars.P;//make_float3( x, 0 );
                   float4 fp_n = make_float4( 1, 0, 0, 0 );  // start derivative at real 1 (see [2]).
                   const float sq_threshold = 2.0f;   // divergence threshold
                   float oscillatingTime = 0.5;//sin(_t / 256.0f );
                   float p = 8.0f;//(5.0f * abs(oscillatingTime)) + 3.0f; //8;
                   float rad = 0.0f;
                   float dist = 0.0f;
                   float d = 1.0;
                   // Iterate to compute f_n and fp_n for the distance estimator.
                   int i = 32;
                   while( i-- )
                   {
                     rad = length(zn);
                     if( rad > sq_threshold )
                     {
                       dist = 0.5f * rad * logf( rad ) / d;
                     }
                     else
                     {
                       float th = atan2( length( make_float3(zn.x, zn.y, 0.0f) ), zn.z );
                       float phi = atan2( zn.y, zn.x );
                       float rado = pow(rad, p);
                       d = pow(rad, p - 1) * (p-1) * d + 1.0;
                       float sint = sin(th * p);
                       zn.x = rado * sint * cos(phi * p);
                       zn.y = rado * sint * sin(phi * p);
                       zn.z = rado * cos(th * p);
                       zn += vars.P;
                     }
                   }
                   return dist;
                   )";

    return mandelbulb_hit_src;
}

Argument Mandelbulb_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int Mandelbulb_SDFOP::argumentSize()
{
    return args.size();
}
