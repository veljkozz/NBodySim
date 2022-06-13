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

// Id Studios
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



// NOTE: DELETE THIS
/*
class vec2 {
	float buff[2];
public:
	float& x = buff[0];
	float& y = buff[1];
	vec2(float x = 0, float y = 0) {
		buff[0] = x;
		buff[1] = y;
	}
	vec2(float* v)
	{
		buff[0] = v[0];
		buff[1] = v[1];
	}

	vec2& operator=(const vec2& v) {
		buff[0] = v.buff[0];
		buff[1] = v.buff[1];
		return *this;
	}
	float& operator[](int i) { if (i <= 1) return buff[i]; else return buff[1]; }

	float* asArr() { return buff; }
	static vec2 add(vec2 v1, vec2 v2) { return vec2(v1.x + v2.x, v1.y + v2.y); }
	static vec2 sub(vec2 v1, vec2 v2) { return vec2(v1.x - v2.x, v1.y - v2.y); }
	static vec2 mult(vec2 v, float scalar) {
		return vec2(v.x * scalar, v.y * scalar);
	}

	static vec2 normalize(vec2 v) {
		return mult(v, invSqrt(vec2(v[0], v[1]).squared()));
	}

	float dot(float* a, float* b) {
		return a[0] * b[0] + a[1] * b[1];
	}

	float squared() {
		return buff[0] * buff[0] + buff[1] * buff[1];
	}

	// Id Studios
	static float invSqrt(float number)
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

};
*/