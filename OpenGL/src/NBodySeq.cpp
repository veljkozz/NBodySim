#include "NBodySeq.h"
#include <chrono>
#include <iostream> 
#include "Params.h"
#include "utils.h"

NBodySeq::NBodySeq(int numParticles) : numParticles(numParticles), tree(this)
{
	positions = new float[2 * numParticles];
	velocities = new float[2 * numParticles];

	distribution = std::uniform_real_distribution<double>(lowbound, bound);
	std::uniform_real_distribution<double> distributionPI(0, 2 * PI);
	for (int i = 0; i < numParticles; i++)
	{
		float angle = distributionPI(generator); // (0, TWO_PI);
		float dist = distribution(generator);
		float mag = 0.001;// dist * 0.002;

		positions[i * 2] = dist * cos(angle);
		positions[i * 2 + 1] = dist * sin(angle);
		velocities[i * 2] = mag * cos(angle + PI / 2);
		velocities[i * 2 + 1] = mag * sin(angle - PI / 2);
	}
}


void NBodySeq::buildQuadTree() {
	float left, right, top, bottom;
	left = right = positions[0]; top = bottom = positions[1];
	for (int i = 1; i < numParticles; i++)
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

	tree.clear();
	QuadTree::Node n(positions[0], positions[1], 0);
	n.left = left; n.right = right; n.bottom = bottom; n.top = top;
	tree.insert(n);
	//tree.root->left = left; tree.root->right = right; tree.root->top = top; tree.root->bottom = bottom;
	for (int i = 1; i < numParticles; i++)
	{
		QuadTree::Node node(positions[i * 2], positions[i * 2 + 1], i);
		tree.insert(node);
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
				else acc = mult(sub(&positions[i * 2], &positions[j * 2]), dt * G * mass / (dSq * sqrt(dSq) + 2));

				velocities[i * 2] -= acc.x;
				velocities[i * 2 + 1] -= acc.y;
				velocities[j * 2] += acc.x;
				velocities[j * 2 + 1] += acc.y;
			}
		}
		// Add black hole in centre?
		//float centerPos[2] = { 0.f, 0.f };
		//float centerMass = 0.001f;
		//dSq = distSquared(&positions[i * 2], centerPos);
		//acc = mult(sub(&positions[i * 2], centerPos), dt * G * mass * centerMass * dSq / (float)sqrt(dSq + 0.0001));
		//velocities[i * 2] += acc.x;
		//velocities[i * 2 + 1] += acc.y;

		positions[i * 2] += velocities[i * 2];
		positions[i * 2 + 1] += velocities[i * 2 + 1];
	}
	buildQuadTree();
}

void NBodySeq::runBarnesHut()
{
	auto start = std::chrono::high_resolution_clock::now();

	buildQuadTree();

	if (DISPLAY_TIMES)
	{
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << "Time for building quad tree:" << duration.count() / 1000 << " milliseconds " << std::endl;

		start = std::chrono::high_resolution_clock::now();

	}
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < numParticles; ++i) {
		tree.forceCalculations(i);

		// Add black hole in centre?
		float centerPos[2] = { 0.f, 0.f };
		float centerMass = 100.f;
		float dSq = distSquared(&positions[i * 2], centerPos);
		Vector acc = mult(sub(&positions[i * 2], centerPos), dt * G * centerMass / (float)sqrt(dSq + 0.0001));
		if (dSq <= 12.f)
			acc = mult(acc, -1);
		
		//velocities[i * 2] -= acc.x;
		//velocities[i * 2 + 1] -= acc.y;

		positions[i * 2] += velocities[i * 2];
		positions[i * 2 + 1] += velocities[i * 2 + 1];
	}
	


	if (DISPLAY_TIMES) {
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << "Time for force cals:" << duration.count() / 1000 << " milliseconds " << std::endl;
	}
	

}

void QuadTree::calcForce(Node* t, int i)
{
	float dSq = distSquared(&sim->positions[i * 2], &sim->positions[t->id * 2]);
	Vector acc;
	if (dSq <= 4 * sim->r * sim->r) {
		acc = Vector(0, 0);
	}
	else acc = mult(sub(&sim->positions[i * 2], &sim->positions[t->id * 2]), sim->dt * sim->G * t->mass / (dSq * sqrt(dSq) + 2));

	sim->velocities[i * 2] -= acc.x;
	sim->velocities[i * 2 + 1] -= acc.y;
	numCalcs++;
}

void QuadTree::calcForceRecursive(Node* t, int i)
{
	if (t == 0) return;
	if (t->leaf && t->id != i)
	{
		calcForce(t, i);
	}
	else {
		float s = t->right - t->left;
		float d = dist(&sim->positions[i], t->pos);
		// if s/d < theta then treat this as a single body
		if (s / d < sim->theta)
		{
			calcForce(t, i);
		}
		else {
			calcForceRecursive(t->nw, i);
			calcForceRecursive(t->ne, i);
			calcForceRecursive(t->sw, i);
			calcForceRecursive(t->se, i);
		}
	}
}