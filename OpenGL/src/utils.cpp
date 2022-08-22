#include "utils.h"
#include <cmath>

float distSquared(float* a, float* b) {
	return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
}

float dist(float* a, float* b)
{
	return sqrt(distSquared(a, b));
}

float add(float* a, float* b) {
	return (a[0] + b[0], a[1] + b[1]);
}

Vector sub(float* a, float* b) {
	return Vector(a[0] - b[0], a[1] - b[1]);
}

Vector mult(float* a, float scalar) {
	return Vector(a[0] * scalar, a[1] * scalar);
}

// Id Studios inverse square root
float invSqrt(float number)
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y = number;
	i = *(long*)&y;                       // evil floating point bit level hacking
	i = 0x5f3759df - (i >> 1);               // what the fuck? 
	y = *(float*)&i;
	y = y * (threehalfs - (x2 * y * y));   // 1st iteration
//	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}

Vector normalize(float* a) {
	return mult(a, invSqrt(Vector(a[0], a[1]).magSquared()));
}
