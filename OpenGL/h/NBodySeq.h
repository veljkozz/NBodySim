#pragma once

#include <random>
#include "Params.h"
class QuadTree;
class NBodySeq
{
public:
	SimulationType simType = DISK_MODEL;
	// Simulation parameters
	const float bound = 18.f, lowbound = 2.5f;

	float r = 0.8; // Radius
	float G = 0.0000001;
	float theta = THETA;
	float dt = 1; // Timestep
public:
	int numParticles;
	float* positions;
	float* velocities;
	float* mass;
	NBodySeq(int numParticles);
	
	void diskModel();

	void buildQuadTree();
	// Brute force implementation
	void runBruteForce();
	
	void runBarnesHut();
	
	~NBodySeq() {
		delete[] positions;
		delete[] velocities;
		delete[] mass;
		delete tree;
	}

	void displayLines();
	int getNumCalcs();
	int getNumNodes();
	const float* getDrawPositions() { return positions; }
private:
	QuadTree* tree;
};