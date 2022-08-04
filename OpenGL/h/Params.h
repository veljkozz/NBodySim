#pragma once


struct Params {
	int numIters = 1;
	bool visualize = false;
	bool display_tree = false;
	bool display_times = true;
	bool display_debug = false;
	int num_particles = 500000;
	float theta = 1.0f;
};

extern Params params;

#define _RUN_CUDA false
#define BRUTEFORCE true



constexpr float PI = 3.1415926535;

