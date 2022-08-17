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

#pragma once

class BarnesHutCUDA
{
public:
	// Simulation parameters
	const float bound = 25.f, lowbound = 1.f;
	float r = 0.8; // Radius
	float G = 0.0000001;
	float theta;
	float dt = 1; // Timestep
	int numParticles;
	int numNodes;

	int step;

	float* h_left;
	float* h_right;
	float* h_bottom;
	float* h_top;

	float* h_mass;
	float* h_x;
	float* h_y;
	float* h_vx;
	float* h_vy;
	float* h_ax;
	float* h_ay;

	int* h_child;
	int* h_start;
	int* h_sorted;
	int* h_count;

	float* d_left;
	float* d_right;
	float* d_bottom;
	float* d_top;

	float* d_mass;
	float* d_x;
	float* d_y;
	float* d_vx;
	float* d_vy;
	float* d_ax;
	float* d_ay;

	int* d_index;
	int* d_child;
	int* d_start;
	int* d_sorted;
	int* d_count;

	int* d_mutex;  //used for locking 

	//cudaEvent_t start, stop; // used for timing

	float* h_output;  //host output array for visualization
	float* d_output;  //device output array for visualization

	float* d_out; // result of reduction operation


public:
	BarnesHutCUDA(int n);
	~BarnesHutCUDA();

	// obrada jedne iteracije simulacije
	void update();
	// inicijalizacija simulacije
	void init();
	// rezultujuce pozicije tela
	const float* getOutput() { return h_output; }
private:
	void diskModel();
};