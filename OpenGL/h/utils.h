#pragma once

struct Vector
{
	float pt[2];
	float& x = pt[0];
	float& y = pt[1];
	Vector(float x, float y) { this->x = x; this->y = y; }
	Vector() { x = 0; y = 0; }
	Vector(const Vector& v) { x = v.x; y = v.y; }
	const Vector& operator=(const Vector& v) { x = v.x; y = v.y; return *this; }
	float magSquared() {
		return x * x + y * y;
	}

	operator float* () { return pt; }
};

float distSquared(float* a, float* b);
float dist(float* a, float* b);
float add(float* a, float* b);
Vector sub(float* a, float* b);
Vector mult(float* a, float scalar);
float invSqrt(float number);