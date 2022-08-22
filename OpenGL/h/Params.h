#pragma once


struct Params {
	int num_particles = 50;
	int numIters = 0;
	bool visualize = true;
	bool display_tree = true;
	bool display_times = false;
	bool display_debug = false;
	float theta = 0.5f;
	bool RUN_CUDA = false;
	bool BRUTEFORCE = false;
	bool BARNES_HUT = true;
};

extern Params params;

constexpr float PI = 3.1415926535;

