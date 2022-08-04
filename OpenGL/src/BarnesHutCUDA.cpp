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

#include "Params.h"
#include "BarnesHutCUDA.h"

#include <random>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.cuh"
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

//int gridSize = 512;
//int blockSize = 256;

BarnesHutCUDA::BarnesHutCUDA(int n)
{
	theta = params.theta;
    numParticles = n;
	numNodes = 2 * n + 12000; // why 12000 ? 

	// allocate host data
	h_left = new float;
	h_right = new float;
	h_bottom = new float;
	h_top = new float;
	h_mass = new float[numNodes];
	h_x = new float[numNodes];
	h_y = new float[numNodes];
	h_vx = new float[numNodes];
	h_vy = new float[numNodes];
	h_ax = new float[numNodes];
	h_ay = new float[numNodes];
	h_child = new int[4 * numNodes];
	h_start = new int[numNodes];
	h_sorted = new int[numNodes];
	h_count = new int[numNodes];
	h_output = new float[2 * numNodes];

	// allocate device data
	gpuErrchk(cudaMalloc((void**)&d_left, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_right, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_bottom, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_top, sizeof(float)));
	gpuErrchk(cudaMemset(d_left, 0, sizeof(float)));
	gpuErrchk(cudaMemset(d_right, 0, sizeof(float)));
	gpuErrchk(cudaMemset(d_bottom, 0, sizeof(float)));
	gpuErrchk(cudaMemset(d_top, 0, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_mass, numNodes * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_x, numNodes * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_y, numNodes * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_vx, numNodes * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_vy, numNodes * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_ax, numNodes * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_ay, numNodes * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_index, sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_child, 4 * numNodes * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_start, numNodes * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_sorted, numNodes * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_count, numNodes * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_mutex, sizeof(int)));

	gpuErrchk(cudaMemset(d_start, -1, numNodes * sizeof(int)));
	gpuErrchk(cudaMemset(d_sorted, 0, numNodes * sizeof(int)));

	int memSize = sizeof(float) * 2 * numParticles;

	gpuErrchk(cudaMalloc((void**)&d_output, 2 * numNodes * sizeof(float)));

}


void BarnesHutCUDA::update()
{
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	ResetArrays(d_mutex, d_x, d_y, d_mass, d_count, d_start, d_sorted, d_child, d_index, d_left, d_right, d_bottom, d_top, numParticles, numNodes);
	ComputeBoundingBox(d_mutex, d_x, d_y, d_left, d_right, d_bottom, d_top, numParticles);
	BuildQuadTree(d_x, d_y, d_mass, d_count, d_start, d_child, d_index, d_left, d_right, d_bottom, d_top, numParticles, numNodes);
	ComputeCentreOfMass(d_x, d_y, d_mass, d_index, numParticles);
	SortParticles(d_count, d_start, d_sorted, d_child, d_index, numParticles);
	CalculateForces(d_x, d_y, d_vx, d_vy, d_ax, d_ay, d_mass, d_sorted, d_child, d_left, d_right, numParticles, G);
	IntegrateParticles(d_x, d_y, d_vx, d_vy, d_ax, d_ay, numParticles, 2, G);
	FillOutputArray(d_x, d_y, d_output, numNodes);

	if (params.display_times)
	{
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	/*if (params.display_times == true) {
		std::cout << "Timestep: " << step << "    " << "Elapsed time: " << elapsedTime << "ms" << std::endl;
	}*/

	

	cudaMemcpy(h_output, d_output, 2 * numNodes * sizeof(float), cudaMemcpyDeviceToHost);

	if (params.display_debug)
	{
		cudaMemcpy(h_left, d_left, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_right, d_right, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_bottom, d_bottom, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_top, d_top, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_child, d_child, 4 * numNodes * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_x, d_x, numNodes * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_y, d_y, numNodes * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_mass, d_mass, numNodes * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_count, d_count, numNodes * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_sorted, d_sorted, numNodes * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_start, d_start, numNodes * sizeof(int), cudaMemcpyDeviceToHost);
		//Debug
	}
	
	step++;
	
}


void BarnesHutCUDA::init()
{
	// set initial mass, position, and velocity of particles
	diskModel();

	// copy data to GPU device
	cudaMemcpy(d_mass, h_mass, 2 * numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, 2 * numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, 2 * numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vx, h_vx, 2 * numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vy, h_vy, 2 * numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ax, h_ax, 2 * numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ay, h_ay, 2 * numParticles * sizeof(float), cudaMemcpyHostToDevice);
	
}

void BarnesHutCUDA::diskModel()
{
	float a = 1.0;
	float pi = 3.14159265;
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(lowbound, bound);
	std::uniform_real_distribution<float> distribution_theta(0.0, 2 * pi);

	// loop through all particles
	for (int i = 0; i < numParticles; i++) {
		float theta = distribution_theta(generator);
		float r = distribution(generator);

		// set mass and position of particle
		if (i == 0) {
			h_mass[i] = 100000.0;
			h_x[i] = 0;
			h_y[i] = 0;
		}
		else {
			h_mass[i] = 1.0;
			h_x[i] = r * cos(theta);
			h_y[i] = r * sin(theta);
		}


		// set velocity of particle
		float rotation = 1;  // 1: clockwise   -1: counter-clockwise 
		float v = 1.0 * sqrt(G * 100000.0 / r);
		if (i == 0) {
			h_vx[0] = 0;
			h_vy[0] = 0;
		}
		else {
			h_vx[i] = rotation * v * sin(theta);
			h_vy[i] = -rotation * v * cos(theta);
		}

		// set acceleration to zero
		h_ax[i] = 0.0;
		h_ay[i] = 0.0;
	}
}

BarnesHutCUDA::~BarnesHutCUDA()
{
	delete h_left;
	delete h_right;
	delete h_bottom;
	delete h_top;
	delete[] h_mass;
	delete[] h_x;
	delete[] h_y;
	delete[] h_vx;
	delete[] h_vy;
	delete[] h_ax;
	delete[] h_ay;
	delete[] h_child;
	delete[] h_start;
	delete[] h_sorted;
	delete[] h_count;
	delete[] h_output;

	gpuErrchk(cudaFree(d_left));
	gpuErrchk(cudaFree(d_right));
	gpuErrchk(cudaFree(d_bottom));
	gpuErrchk(cudaFree(d_top));

	gpuErrchk(cudaFree(d_mass));
	gpuErrchk(cudaFree(d_x));
	gpuErrchk(cudaFree(d_y));
	gpuErrchk(cudaFree(d_vx));
	gpuErrchk(cudaFree(d_vy));
	gpuErrchk(cudaFree(d_ax));
	gpuErrchk(cudaFree(d_ay));

	gpuErrchk(cudaFree(d_index));
	gpuErrchk(cudaFree(d_child));
	gpuErrchk(cudaFree(d_start));
	gpuErrchk(cudaFree(d_sorted));
	gpuErrchk(cudaFree(d_count));

	gpuErrchk(cudaFree(d_mutex));

	gpuErrchk(cudaFree(d_output));

	gpuErrchk(cudaFree(d_out));

	cudaDeviceSynchronize();
}
