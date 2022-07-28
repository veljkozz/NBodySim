/*
Copyright 2016 James Sandham

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
*/

#ifndef __KERNELS_H__
#define __KERNELS_H__


const unsigned int gridSize = 512;
const unsigned int blockSize = 256;

// THESE ARE MINE

void BoundingBox(int* mutex, float* x, float* y, float* left, float* right, float* bottom, float* top, int n);
__global__ void bounding_box_kernel(int* mutex, float* left_in, float* right_in, float* bottom_in, float* top_in, float* left, float* right, float* bottom, float* top, int n);


//
void SetDrawArray(float* ptr, float* x, float* y, int n);
void ResetArrays(int* mutex, float* x, float* y, float* mass, int* count, int* start, int* sorted, int* child, int* index, float* left, float* right, float* bottom, float* top, int n, int m);
void ComputeBoundingBox(int* mutex, float* x, float* y, float* left, float* right, float* bottom, float* top, int n);
void BuildQuadTree(float* x, float* y, float* mass, int* count, int* start, int* child, int* index, float* left, float* right, float* bottom, float* top, int n, int m);
void ComputeCentreOfMass(float* x, float* y, float* mass, int* index, int n);
void SortParticles(int* count, int* start, int* sorted, int* child, int* index, int n);
void CalculateForces(float* x, float* y, float* vx, float* vy, float* ax, float* ay, float* mass, int* sorted, int* child, float* left, float* right, int n, float g);
void IntegrateParticles(float* x, float* y, float* vx, float* vy, float* ax, float* ay, int n, float dt, float d);
void FillOutputArray(float* x, float* y, float* out, int n);

__global__ void set_draw_array_kernel(float* ptr, float* x, float* y, int n);
__global__ void reset_arrays_kernel(int* mutex, float* x, float* y, float* mass, int* count, int* start, int* sorted, int* child, int* index, float* left, float* right, float* bottom, float* top, int n, int m);
__global__ void compute_bounding_box_kernel(int* mutex, float* x, float* y, float* left, float* right, float* bottom, float* top, int n);
__global__ void build_tree_kernel(float* x, float* y, float* mass, int* count, int* start, int* child, int* index, float* left, float* right, float* bottom, float* top, int n, int m);
__global__ void centre_of_mass_kernel(float* x, float* y, float* mass, int* index, int n);
__global__ void sort_kernel(int* count, int* start, int* sorted, int* child, int* index, int n);
__global__ void compute_forces_kernel(float* x, float* y, float* vx, float* vy, float* ax, float* ay, float* mass, int* sorted, int* child, float* left, float* right, int n, float g);
__global__ void update_kernel(float* x, float* y, float* vx, float* vy, float* ax, float* ay, int n, float dt, float d);
__global__ void copy_kernel(float* x, float* y, float* out, int n);

__global__ void test_kernel(float* x, float* y, float* left, float* right, float* bottom, float* top, int n);
void Test(float* x, float* y, float* left, float* right, float* bottom, float* top, int n);

#endif