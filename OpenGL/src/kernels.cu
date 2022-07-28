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

#include "kernels.cuh"

#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <cmath>

// ==========================================================================================
// CUDA ERROR CHECKING CODE
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) getchar();
	}
}

__device__ const int warp = 32;
__device__ const int stackSize = 64;
__device__ const float eps2 = 0.025;
__device__ const float theta = 0.5;

//const unsigned int gridSize = 512;
//const unsigned int blockSize = 256;

int gridS = 512;
// MY KERNELS

void BoundingBox(int* mutex, float* x, float* y, float* left, float* right, float* bottom, float* top, int n){
	compute_bounding_box_kernel << < ceil(n/blockSize), blockSize >> > (mutex, x, y, left, right, bottom, top, n);
}


void SetDrawArray(float* ptr, float* x, float* y, int n)
{
	set_draw_array_kernel <<< gridSize, blockSize >> > (ptr, x, y, n);
}


void ResetArrays(int* mutex, float* x, float* y, float* mass, int* count, int* start, int* sorted, int* child, int* index, float* left, float* right, float* bottom, float* top, int n, int m)
{
	reset_arrays_kernel <<< gridSize, blockSize >> > (mutex, x, y, mass, count, start, sorted, child, index, left, right, bottom, top, n, m);
}


void ComputeBoundingBox(int* mutex, float* x, float* y, float* left, float* right, float* bottom, float* top, int n)
{
	compute_bounding_box_kernel <<< gridSize, blockSize >> > (mutex, x, y, left, right, bottom, top, n);
}


void BuildQuadTree(float* x, float* y, float* mass, int* count, int* start, int* child, int* index, float* left, float* right, float* bottom, float* top, int n, int m)
{
	build_tree_kernel << < gridSize, blockSize >> > (x, y, mass, count, start, child, index, left, right, bottom, top, n, m);
}


void ComputeCentreOfMass(float* x, float* y, float* mass, int* index, int n)
{
	centre_of_mass_kernel << <gridSize, blockSize >> > (x, y, mass, index, n);
}


void SortParticles(int* count, int* start, int* sorted, int* child, int* index, int n)
{
	sort_kernel << < gridSize, blockSize >> > (count, start, sorted, child, index, n);
}


void CalculateForces(float* x, float* y, float* vx, float* vy, float* ax, float* ay, float* mass, int* sorted, int* child, float* left, float* right, int n, float g)
{
	compute_forces_kernel << < gridSize, blockSize >> > (x, y, vx, vy, ax, ay, mass, sorted, child, left, right, n, g);
}


void IntegrateParticles(float* x, float* y, float* vx, float* vy, float* ax, float* ay, int n, float dt, float d)
{
	update_kernel << <gridSize, blockSize >> > (x, y, vx, vy, ax, ay, n, dt, d);
}


void FillOutputArray(float* x, float* y, float* out, int n)
{
	copy_kernel << <gridSize, blockSize >> > (x, y, out, n);
}

__global__ void set_draw_array_kernel(float* ptr, float* x, float* y, int n)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index < n) {
		ptr[2 * index] = x[index];
		ptr[2 * index + 1] = y[index];
	}
}


__global__ void reset_arrays_kernel(int* mutex, float* x, float* y, float* mass, int* count, int* start, int* sorted, int* child, int* index, float* left, float* right, float* bottom, float* top, int n, int m)
{
	int bodyIndex = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	int offset = 0;

	// reset quadtree arrays
	while (bodyIndex + offset < m) {
#pragma unroll 4
		for (int i = 0; i < 4; i++) {
			child[(bodyIndex + offset) * 4 + i] = -1;
		}
		if (bodyIndex + offset < n) {
			count[bodyIndex + offset] = 1;
		}
		else {
			x[bodyIndex + offset] = 0;
			y[bodyIndex + offset] = 0;
			mass[bodyIndex + offset] = 0;
			count[bodyIndex + offset] = 0;
		}
		start[bodyIndex + offset] = -1;
		sorted[bodyIndex + offset] = 0;
		offset += stride;
	}

	if (bodyIndex == 0) {
		*mutex = 0;
		*index = n;
		*left = 0;
		*right = 0;
		*bottom = 0;
		*top = 0;
	}
}

__global__ void bounding_box_kernel(int* mutex, float* left_in, float* right_in, float* bottom_in, float * top_in, float* left, float* right, float* bottom, float* top, int n)
{
	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tid = threadIdx.x;

	__shared__ float left_cache[blockSize];
	__shared__ float right_cache[blockSize];
	__shared__ float bottom_cache[blockSize];
	__shared__ float top_cache[blockSize];
	if (id < n)
	{
		float x_min = left_in[id];
		float x_max = right_in[id];
		float y_min = bottom_in[id];
		float y_max = top_in[id];

		left_cache[tid] = x_min;
		right_cache[tid] = x_max;
		bottom_cache[tid] = y_min;
		top_cache[tid] = y_max;

		//if (tid == 0) printf("%d \n", blockIdx.x);

		__syncthreads();

		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
		{
			if (tid < s)
			{
				left_cache[tid] = fminf(left_cache[tid], left_cache[tid + s]);
				right_cache[tid] = fmaxf(left_cache[tid], right_cache[tid + s]);
				bottom_cache[tid] = fminf(bottom_cache[tid], bottom_cache[tid + s]);
				top_cache[tid] = fmaxf(top_cache[tid], top_cache[tid + s]);
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			left[blockIdx.x] = left_cache[0];
			right[blockIdx.x] = right_cache[0];
			top[blockIdx.x] = top_cache[0];
			bottom[blockIdx.x] = bottom_cache[0];
		}
	}
	

}

__global__ void compute_bounding_box_kernel(int* mutex, float* x, float* y, float* left, float* right, float* bottom, float* top, int n)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	

	__shared__ float left_cache[blockSize];
	__shared__ float right_cache[blockSize];
	__shared__ float bottom_cache[blockSize];
	__shared__ float top_cache[blockSize];
	if (index < n)
	{
		float x_min = x[index];
		float x_max = x[index];
		float y_min = y[index];
		float y_max = y[index];
		int offset = stride;
		while (index + offset < n) {
			x_min = fminf(x_min, x[index + offset]);
			x_max = fmaxf(x_max, x[index + offset]);
			y_min = fminf(y_min, y[index + offset]);
			y_max = fmaxf(y_max, y[index + offset]);
			offset += stride;
		}

		left_cache[threadIdx.x] = x_min;
		right_cache[threadIdx.x] = x_max;
		bottom_cache[threadIdx.x] = y_min;
		top_cache[threadIdx.x] = y_max;

		__syncthreads();

		// assumes blockDim.x is a power of 2!
		int i = blockDim.x / 2;
		while (i != 0) {
			if (threadIdx.x < i) {
				left_cache[threadIdx.x] = fminf(left_cache[threadIdx.x], left_cache[threadIdx.x + i]);
				right_cache[threadIdx.x] = fmaxf(right_cache[threadIdx.x], right_cache[threadIdx.x + i]);
				bottom_cache[threadIdx.x] = fminf(bottom_cache[threadIdx.x], bottom_cache[threadIdx.x + i]);
				top_cache[threadIdx.x] = fmaxf(top_cache[threadIdx.x], top_cache[threadIdx.x + i]);
			}
			__syncthreads();
			i /= 2;
		}

		if (threadIdx.x == 0) {
			while (atomicCAS(mutex, 0, 1) != 0); // lock
			*left = fminf(*left, left_cache[0]);
			*right = fmaxf(*right, right_cache[0]);
			*bottom = fminf(*bottom, bottom_cache[0]);
			*top = fmaxf(*top, top_cache[0]);
			atomicExch(mutex, 0); // unlock
		}
	}
}

__global__ void build_tree_kernel(float* x, float* y, float* mass, int* count, int* start, int* child, int* index, float* left, float* right, float* bottom, float* top, int n, int m)
{
	int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int offset = 0;
	bool newBody = true;

	// build quadtree
	float l;
	float r;
	float b;
	float t;
	int childPath;
	int temp;
	offset = 0;
	while ((bodyIndex + offset) < n) {

		if (newBody) {
			newBody = false;

			l = *left;
			r = *right;
			b = *bottom;
			t = *top;

			temp = 0;
			childPath = 0;
			if (x[bodyIndex + offset] < 0.5 * (l + r)) {
				childPath += 1;
				r = 0.5 * (l + r);
			}
			else {
				l = 0.5 * (l + r);
			}
			if (y[bodyIndex + offset] < 0.5 * (b + t)) {
				childPath += 2;
				t = 0.5 * (t + b);
			}
			else {
				b = 0.5 * (t + b);
			}
		}
		int childIndex = child[temp * 4 + childPath];

		// traverse tree until we hit leaf node
		while (childIndex >= n) {
			temp = childIndex;
			childPath = 0;
			if (x[bodyIndex + offset] < 0.5 * (l + r)) {
				childPath += 1;
				r = 0.5 * (l + r);
			}
			else {
				l = 0.5 * (l + r);
			}
			if (y[bodyIndex + offset] < 0.5 * (b + t)) {
				childPath += 2;
				t = 0.5 * (t + b);
			}
			else {
				b = 0.5 * (t + b);
			}

			atomicAdd(&x[temp], mass[bodyIndex + offset] * x[bodyIndex + offset]);
			atomicAdd(&y[temp], mass[bodyIndex + offset] * y[bodyIndex + offset]);
			atomicAdd(&mass[temp], mass[bodyIndex + offset]);
			atomicAdd(&count[temp], 1);
			childIndex = child[4 * temp + childPath];
		}


		if (childIndex != -2) {
			int locked = temp * 4 + childPath;
			if (atomicCAS(&child[locked], childIndex, -2) == childIndex) {
				if (childIndex == -1) {
					child[locked] = bodyIndex + offset;
				}
				else {
					//int patch = 2*n;
					int patch = 4 * n;
					while (childIndex >= 0 && childIndex < n) {

						int cell = atomicAdd(index, 1);
						patch = min(patch, cell);
						if (patch != cell) {
							child[4 * temp + childPath] = cell;
						}

						// insert old particle
						childPath = 0;
						if (x[childIndex] < 0.5 * (l + r)) {
							childPath += 1;
						}
						if (y[childIndex] < 0.5 * (b + t)) {
							childPath += 2;
						}

						if (true) {
							// if(cell >= 2*n){
							if (cell >= m) {
								printf("%s\n", "error cell index is too large!!");
								printf("cell: %d\n", cell);
							}
						}
						x[cell] += mass[childIndex] * x[childIndex];
						y[cell] += mass[childIndex] * y[childIndex];
						mass[cell] += mass[childIndex];
						count[cell] += count[childIndex];
						child[4 * cell + childPath] = childIndex;

						start[cell] = -1;


						// insert new particle
						temp = cell;
						childPath = 0;
						if (x[bodyIndex + offset] < 0.5 * (l + r)) {
							childPath += 1;
							r = 0.5 * (l + r);
						}
						else {
							l = 0.5 * (l + r);
						}
						if (y[bodyIndex + offset] < 0.5 * (b + t)) {
							childPath += 2;
							t = 0.5 * (t + b);
						}
						else {
							b = 0.5 * (t + b);
						}
						x[cell] += mass[bodyIndex + offset] * x[bodyIndex + offset];
						y[cell] += mass[bodyIndex + offset] * y[bodyIndex + offset];
						mass[cell] += mass[bodyIndex + offset];
						count[cell] += count[bodyIndex + offset];
						childIndex = child[4 * temp + childPath];
					}

					child[4 * temp + childPath] = bodyIndex + offset;

					__threadfence();  // we have been writing to global memory arrays (child, x, y, mass) thus need to fence

					child[locked] = patch;
				}

				// __threadfence(); // we have been writing to global memory arrays (child, x, y, mass) thus need to fence

				offset += stride;
				newBody = true;
			}

		}

		__syncthreads(); // not strictly needed 
	}
}

__global__ void centre_of_mass_kernel(float* x, float* y, float* mass, int* index, int n)
{
	int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int offset = 0;

	bodyIndex += n;
	while (bodyIndex + offset < *index) {
		x[bodyIndex + offset] /= mass[bodyIndex + offset];
		y[bodyIndex + offset] /= mass[bodyIndex + offset];

		offset += stride;
	}
}

__global__ void sort_kernel(int* count, int* start, int* sorted, int* child, int* index, int n)
{
	int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int offset = 0;

	int s = 0;
	if (threadIdx.x == 0) {
		for (int i = 0; i < 4; i++) {
			int node = child[i];

			if (node >= n) {  // not a leaf node
				start[node] = s;
				s += count[node];
			}
			else if (node >= 0) {  // leaf node
				sorted[s] = node;
				s++;
			}
		}
	}

	int cell = n + bodyIndex;
	int ind = *index;
	while ((cell + offset) < ind) {
		s = start[cell + offset];

		if (s >= 0) {

			for (int i = 0; i < 4; i++) {
				int node = child[4 * (cell + offset) + i];

				if (node >= n) {  // not a leaf node
					start[node] = s;
					s += count[node];
				}
				else if (node >= 0) {  // leaf node
					sorted[s] = node;
					s++;
				}
			}
			offset += stride;
		}
	}
}

__global__ void compute_forces_kernel(float* x, float* y, float* vx, float* vy, float* ax, float* ay, float* mass, int* sorted, int* child, float* left, float* right, int n, float g)
{
	int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int offset = 0;

	__shared__ float depth[stackSize * blockSize / warp];
	__shared__ int stack[stackSize * blockSize / warp];  // stack controled by one thread per warp 

	float radius = 0.5 * (*right - (*left));

	// need this in case some of the first four entries of child are -1 (otherwise jj = 3)
	int jj = -1;
	for (int i = 0; i < 4; i++) {
		if (child[i] != -1) {
			jj++;
		}
	}

	int counter = threadIdx.x % warp;
	int stackStartIndex = stackSize * (threadIdx.x / warp);
	while (bodyIndex + offset < n) {
		int sortedIndex = sorted[bodyIndex + offset];

		float pos_x = x[sortedIndex];
		float pos_y = y[sortedIndex];
		float acc_x = 0;
		float acc_y = 0;

		// initialize stack
		int top = jj + stackStartIndex;
		if (counter == 0) {
			int temp = 0;
			for (int i = 0; i < 4; i++) {
				if (child[i] != -1) {
					stack[stackStartIndex + temp] = child[i];
					depth[stackStartIndex + temp] = radius * radius / theta;
					temp++;
				}
				// if(child[i] == -1){
				// 	printf("%s %d %d %d %d %s %d\n", "THROW ERROR!!!!", child[0], child[1], child[2], child[3], "top: ",top);
				// }
				// else{
				// 	stack[stackStartIndex + temp] = child[i];
				// 	depth[stackStartIndex + temp] = radius*radius/theta;
				// 	temp++;	
				// }
			}
		}

		__syncthreads();

		// while stack is not empty
		while (top >= stackStartIndex) {
			int node = stack[top];
			float dp = 0.25 * depth[top];
			// float dp = depth[top];
			for (int i = 0; i < 4; i++) {
				int ch = child[4 * node + i];

				//__threadfence();

				if (ch >= 0) {
					float dx = x[ch] - pos_x;
					float dy = y[ch] - pos_y;
					float r = dx * dx + dy * dy + eps2;
					if (ch < n /*is leaf node*/ || __all(dp <= r)/*meets criterion*/) {
						r = rsqrt(r);
						float f = mass[ch] * r * r * r;

						acc_x += f * dx;
						acc_y += f * dy;
					}
					else {
						if (counter == 0) {
							stack[top] = ch;
							depth[top] = dp;
							// depth[top] = 0.25*dp;
						}
						top++;
						//__threadfence();
					}
				}
			}

			top--;
		}

		ax[sortedIndex] = acc_x;
		ay[sortedIndex] = acc_y;

		offset += stride;

		__syncthreads();
	}
}



// __global__ void compute_forces_kernel(float* x, float *y, float *vx, float *vy, float *ax, float *ay, float *mass, int *sorted, int *child, float *left, float *right, int n, float g)
// {
// 	int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
// 	int stride = blockDim.x*gridDim.x;
// 	int offset = 0;

// 	__shared__ float depth[stackSize*blockSize/warp]; 
// 	__shared__ int stack[stackSize*blockSize/warp];  // stack controled by one thread per warp 

// 	int counter = threadIdx.x % warp;
// 	int stackStartIndex = stackSize*(threadIdx.x / warp);
// 	while(bodyIndex + offset < n){
// 		int sortedIndex = sorted[bodyIndex + offset];

// 		float pos_x = x[sortedIndex];
// 		float pos_y = y[sortedIndex];
// 		float acc_x = 0;
// 		float acc_y = 0; 

// 		// initialize stack
// 		int top = 3 + stackStartIndex;
// 		float radius = 0.5*(*right - (*left));
// 		if(counter == 0){
// #pragma unroll 4
// 			for(int i=0;i<4;i++){
// 				if(child[i] == -1){
// 					printf("%s\n", "THROW ERROR!!!!");
// 				}
// 				stack[stackStartIndex + i] = child[i];
// 				depth[stackStartIndex + i] = radius;
// 			}
// 		}

// 		__syncthreads();

// 		// while stack is not empty
// 		while(top >= stackStartIndex){
// 			int node = stack[top];
// 			float dp = 0.5*depth[top];
// 			// float dp = depth[top];
// 			for(int i=0;i<4;i++){
// 				int ch = child[4*node + i];

// 				//__threadfence();

// 				if(ch >= 0){
// 					float dx = x[ch] - pos_x;
//   					float dy = y[ch] - pos_y;
//     				//float r = sqrt(dx*dx + dy*dy + eps2);
//     				float r = rsqrt(dx*dx + dy*dy + eps2);
//     				if(ch < n /*is leaf node*/ || __all(dp*r <= theta)/*meets criterion*/){ 
//     					//float f = mass[ch]/(r*r*r);
//     					float f = mass[ch] * r * r * r;

//     					acc_x += f*dx;
//     					acc_y += f*dy;
// 					}
// 					else{
// 						if(counter == 0){
// 							stack[top] = ch;
// 							depth[top] = dp;
// 							// depth[top] = 0.5*dp;
// 						}
// 						top++;
// 						//__threadfence();
// 					}
// 				}
// 			}

// 			top--;
// 		}

// 		ax[sortedIndex] = acc_x;
//    		ay[sortedIndex] = acc_y;

//    		offset += stride;

//    		__syncthreads();
//    	}
// }



__global__ void update_kernel(float* x, float* y, float* vx, float* vy, float* ax, float* ay, int n, float dt, float d)
{
	int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int offset = 0;

	while (bodyIndex + offset < n) {
		vx[bodyIndex + offset] += dt * ax[bodyIndex + offset] * d;
		vy[bodyIndex + offset] += dt * ay[bodyIndex + offset] * d;

		x[bodyIndex + offset] += dt * vx[bodyIndex + offset];
		y[bodyIndex + offset] += dt * vy[bodyIndex + offset];

		offset += stride;
	}
}



__global__ void copy_kernel(float* x, float* y, float* out, int n)
{
	int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int offset = 0;

	while (bodyIndex + offset < n) {
		out[2 * (bodyIndex + offset)] = x[bodyIndex + offset];
		out[2 * (bodyIndex + offset) + 1] = y[bodyIndex + offset];

		offset += stride;
	}
}


void Test(float* x, float* y, float* left, float* right, float* bottom, float* top, int n)
{
	test_kernel << <gridSize, blockSize >> > (x, y, left, right, bottom, top, n);
}

__global__ void test_kernel(float* x, float* y, float* left, float* right, float * bottom, float* top, int n)
{
	int tid = threadIdx.x;
	float minX = x[0], maxX = x[0], minY = y[0], maxY = y[0];
	if (tid == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < n; i++)
		{
			minX = fminf(minX, x[i]);
			maxX = fmaxf(maxX, x[i]);
			minY = fminf(minY, y[i]);
			maxY = fmaxf(maxY, y[i]);
		}
		*left = minX; *right = maxX;
		*bottom = minY; *top = maxY;
		
	}
}