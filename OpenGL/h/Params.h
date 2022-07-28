#pragma once


struct Params {
	bool visualize = true;
	bool display_tree = true;
	bool display_times = false;
	bool display_debug = false;
	int num_particles = 12;
	float theta = 0.8f;
};

extern Params params;

#define _RUN_CUDA false
//#define DISPLAY_TREE true
#define BRUTEFORCE true



constexpr float PI = 3.1415926535;

