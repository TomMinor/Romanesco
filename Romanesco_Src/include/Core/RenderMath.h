#ifndef RENDERMATH_H
#define RENDERMATH_H

#include <algorithm>

#define SMALL_NUMBER 0.0000000001

template <typename T>
T clamp(const T& n, const T& lower, const T& upper) {
  return std::max(lower, std::min(n, upper));
}

float FInterpTo( float Current, float Target, float DeltaTime, float InterpSpeed );

inline float degrees(float _radians)
{
    return (_radians * (180.0f / M_PI));
}

inline float radians(float _degrees)
{
    return (_degrees * (M_PI / 180.0f));
}

#endif // RENDERMATH_H

