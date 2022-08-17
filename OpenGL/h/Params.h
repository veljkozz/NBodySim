#pragma once


struct Params {
	int num_particles = 100000;
	int numIters = 10000000;
	bool visualize = true;
	bool display_tree = false;
	bool display_times = false;
	bool display_debug = false;
	float theta = 1.0f;
};

extern Params params;

#define _RUN_CUDA true
#define BRUTEFORCE false



constexpr float PI = 3.1415926535;

