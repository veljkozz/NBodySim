#include "NBodySeq.h"
#include <chrono>
#include <iostream> 
#include <thread>
#include "Params.h"

#include "Quadtree.h"
#include "utils.h"

NBodySeq::NBodySeq(int numParticles) : numParticles(numParticles)
{
	theta = params.theta;
	positions = new float[2 * numParticles];
	velocities = new float[2 * numParticles];
	mass = new float[numParticles];
	tree = new QuadTree(this);

	diskModel();
//	// My random model
//	std::default_random_engine generator;
//	std::uniform_real_distribution<double> distribution;
//	distribution = std::uniform_real_distribution<double>(lowbound, bound);
//	std::uniform_real_distribution<double> distributionPI(0, 2 * PI);
//	for (int i = 0; i < numParticles; i++)
//	{
//		float angle = distributionPI(generator); // (0, TWO_PI);
//		float dist = distribution(generator);
//		float mag = 0.001;// dist * 0.002;
//
//		positions[i * 2] = dist * cos(angle);
//		positions[i * 2 + 1] = dist * sin(angle);
//		velocities[i * 2] = mag * cos(angle + PI / 2);
//		velocities[i * 2 + 1] = mag * sin(angle - PI / 2);
//		mass[i] = 1;
//	}
//}
}


void NBodySeq::diskModel()
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
			mass[i] = 100000;
			positions[i*2] = 0;
			positions[i*2 + 1] = 0;
		}
		else {
			mass[i] = 1.0;
			positions[i * 2] = r * cos(theta);
			positions[i * 2 + 1] = r * sin(theta);
		}

		// set velocity of particle
		float rotation = 1;  // 1: clockwise   -1: counter-clockwise 
		float v = 1.0 * sqrt(G * 100000.0 / r);
		if (i == 0) {
			velocities[0] = 0;
			velocities[1] = 0;
		}
		else {
			velocities[i*2] = rotation * v * sin(theta);
			velocities[i*2 + 1] = -rotation * v * cos(theta);
		}
	}

}

void NBodySeq::buildQuadTree() {
	float left, right, top, bottom;
	left = right = positions[0]; top = bottom = positions[1];
	for (int i = 0; i < numParticles; i++)
	{
		left = std::min(positions[i * 2], left);
		right = std::max(positions[i * 2], right);
		top = std::max(positions[i * 2 + 1], top);
		bottom = std::min(positions[i * 2 + 1], bottom);
	}

	if (right - left > top - bottom)
	{
		top = bottom + (right - left);
	}
	else {
		right = left + (top - bottom);
	}

	tree->clear();
	QuadTree::Node n(positions[0], positions[1], 0);
	n.left = left; n.right = right; n.bottom = bottom; n.top = top;
	n.mass = mass[0];
	tree->insert(n);

	for (int i = 1; i < numParticles; i++)
	{
		QuadTree::Node node(positions[i * 2], positions[i * 2 + 1], i);
		node.mass = mass[i];
		tree->insert(node);
	}

}

void NBodySeq::runBruteForce() {


	for (int i = 0; i < numParticles; i++)
	{
		Vector acc;
		float dSq;
		for (int j = i + 1; j < numParticles; j++)
		{
			if (i != j)
			{
				dSq = distSquared(&positions[i * 2], &positions[j * 2]);
				if (dSq <= 4 * r * r) {
					acc = Vector(0, 0);
				}
				else acc = mult(sub(&positions[i * 2], &positions[j * 2]), dt * G * mass[j] * mass[i] / (dSq * sqrt(dSq) + 2));

				velocities[i * 2] -= acc.x / mass[i];
				velocities[i * 2 + 1] -= acc.y / mass[i];
				velocities[j * 2] += acc.x / mass[j];
				velocities[j * 2 + 1] += acc.y / mass[j];
			}
		}
		
		if (i != 0)
		{
			positions[i * 2] += velocities[i * 2];
			positions[i * 2 + 1] += velocities[i * 2 + 1];
		}
		
	}
	buildQuadTree();
}

void NBodySeq::runBarnesHut()
{

	buildQuadTree();

	for (int i = 1; i < numParticles; ++i) {
		tree->forceCalculations(i);

		positions[i * 2] += velocities[i * 2];
		positions[i * 2 + 1] += velocities[i * 2 + 1];

		using namespace std::chrono_literals;

	}
	

}


void NBodySeq::displayLines() { tree->displayLines(); }
int NBodySeq::getNumCalcs() { return tree->getNumCalcs(); }

int NBodySeq::getNumNodes() { return tree->getNumNodes(); }