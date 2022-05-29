#pragma once

#include <random>
#include <cmath>

#define PI 3.14159265359

class NBodySeq
{
private:
	// Simulation parameters
	const float bound = 15.f, lowbound = 3.f;

	float r = 1; // Radius
	const float mass = 1;
	float G = 0.00001;
	float theta = 2;
	float dt = 1; // Timestep
public:
	int numCalcs;
	NBodySeq(int numParticles);

	struct Node
	{
		bool leaf = true;
		float pos[2];
		float left, right, top, bottom;
		float mass = 1;
		Node* nw = 0, * ne = 0, * sw = 0, * se = 0;
		int id;
		Node(float l, float r, float t, float b) : left(l), right(r), top(t), bottom(b) {};
		Node(float x, float y, int id) { pos[0] = x; pos[1] = y; this->id = id; }
	};
	
	class QuadTree {
	public:
		
		Node* root;
		void insert(Node& n);
		void insertRecursive(Node* t, Node& n);
		void insertRecursiveChoice(Node* t, Node n);
		void updateParentChild(Node* parent, Node* child);
		void updateMass(Node* t);
		void displayLinesRecursive(Node* root);
		void displayLines();
		void clear();
		void deleteRecursive(Node* t);
		~QuadTree();
	};
	QuadTree tree;

	void buildQuadTree();
	// Brute force implementation
	void runBruteForce();

	
	void runBarnesHut();
	void calcForce(Node* t, int i);
	void calcForceRecursive(Node* t, int i);

	~NBodySeq() {
		delete[] positions;
		delete[] velocities;
	}

	const float* getPositions() { return positions; }
private:
	int numParticles;

	
	float* positions;
	float* velocities;

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution;

	
	
};