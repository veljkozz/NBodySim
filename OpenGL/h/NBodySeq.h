#pragma once

#include <random>
#include "QuadTree.h"
#include "Params.h"
#define PI 3.14159265359

class NBodySeq
{
public:
	// Simulation parameters
	const float bound = 15.f, lowbound = 3.f;

	float r = 0.8; // Radius
	const float mass = 1;
	float G = 0.00001;
	float theta = THETA;
	float dt = 1; // Timestep
public:
	int numParticles;
	float* positions;
	float* velocities;
	NBodySeq(int numParticles);
	
	void buildQuadTree();
	// Brute force implementation
	void runBruteForce();
	
	void runBarnesHut();
	
	~NBodySeq() {
		delete[] positions;
		delete[] velocities;
	}

	void displayLines() { tree.displayLines(); }
	int getNumCalcs() { return tree.getNumCalcs(); }
	const float* getPositions() { return positions; }
private:
	QuadTree tree;
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution;

	
	
};