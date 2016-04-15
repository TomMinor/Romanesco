#ifndef RENDERMATH_H
#define RENDERMATH_H

#include <algorithm>

#define SMALL_NUMBER 0.0000000001

template <typename T>
T clamp(const T& n, const T& lower, const T& upper) {
  return std::max(lower, std::min(n, upper));
}

float FInterpTo( float Current, float Target, float DeltaTime, float InterpSpeed );

#endif // RENDERMATH_H

