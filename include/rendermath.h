#ifndef RENDERMATH_H
#define RENDERMATH_H

#define SMALL_NUMBER 0.0000000001

template <typename T>
T clamp(const T& n, const T& lower, const T& upper) {
  return std::max(lower, std::min(n, upper));
}

float FInterpTo( float Current, float Target, float DeltaTime, float InterpSpeed )
{
  // If no interp speed, jump to target value
  if( InterpSpeed == 0.f )
  {
    return Target;
  }

  // Distance to reach
  const float Dist = Target - Current;

  // If distance is too small, just set the desired location
  if( Dist*Dist < SMALL_NUMBER )
  {
    return Target;
  }

  // Delta Move, Clamp so we do not over shoot.
  const float DeltaMove = Dist * clamp<float>(DeltaTime * InterpSpeed, 0.f, 1.f);

  return Current + DeltaMove;
}

#endif // RENDERMATH_H

