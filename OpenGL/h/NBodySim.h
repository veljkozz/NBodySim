#pragma once

#include <random>
#include <cmath>
#include "utils.h"

#define PI 3.14

class NBodySim
{
public:
	NBodySim(int numParticles) : numParticles(numParticles)
	{
		positions = new float[2 * numParticles];
		velocities = new float[2 * numParticles];
		r = 2;
		G = 1; // Gravitational constant
		dt = 0.01;

		distribution = std::uniform_real_distribution<double>(0, bound);
		std::uniform_real_distribution<double> distributionPI(0, 2*3.14);
		for (int i = 0; i < numParticles; i++)
		{
			float angle = distributionPI(generator); // (0, TWO_PI);
			float dist = distribution(generator);
			float dist2 = distribution(generator);
			float mag = dist * 0.02;

			positions[i * 2] = dist * cos(angle);
			positions[i * 2 + 1] = dist * sin(angle);
			velocities[i * 2] = mag * cos(angle + PI / 2);
			velocities[i * 2 + 1] = mag * sin(angle + PI / 2);
		}
	}

	// north west: nw,..
	float left, right, top, bottom;
	
	class Node
	{
	public:
		bool leaf = true;
		Node* parent = 0;
		float pos[2];
		float left, right, top, bottom;
		float mass;
		Node* nw = 0,* ne = 0,* sw = 0,* se = 0;
		Node(float l, float r, float t, float b) : left(l), right(r), top(t), bottom(b) {};
		Node(float x, float y) { pos[0] = x; pos[1] = y; }
	};
	class QuadTree {
	public:
		Node* root;
		void insert(Node& n)
		{
			if (root)
				insertRecursive(root, n);
			else root = new Node(n);
		}

		void insertRecursive(Node* t, Node& n)
		{
			if (!t->leaf)
			{
				insertRecursiveChoice(t, n);
			}
			else
			{
				Node tn = Node(*t);
				insertRecursiveChoice(t, n);
				insertRecursiveChoice(t, tn);
				t->leaf = false;
			}
		}

		void insertRecursiveChoice(Node* t, Node n)
		{
			float midx = (t->left + t->right) / 2.f;
			float midy = (t->top + t->bottom) / 2.f;
			if (n.pos[0] < midx && n.pos[1] > midy)
			{
				n.left = t->left; n.right = midx; n.top = t->top; n.bottom = midy;
				if (t->nw) insertRecursive(t->nw, n);
				else t->nw = new Node(n);
			}
			else if (n.pos[0] > midx && n.pos[1] > midy)
			{
				n.left = midx; n.right = t->right; n.top = t->top; n.bottom = midy;
				if (t->ne) insertRecursive(t->ne, n);
				else t->ne = new Node(n);
			}
			else if (n.pos[0] < midx && n.pos[1] < midy)
			{
				n.left = t->left; n.right = midx; n.top = midy; n.bottom = t->bottom;
				if (t->sw) insertRecursive(t->sw, n);
				else t->sw = new Node(n);
			}
			else if (n.pos[0] > midx && n.pos[1] < midy)
			{
				n.left = midx; n.right = t->right; n.top = midy; n.bottom = t->bottom;
				if (t->se) insertRecursive(t->se, n);
				else t->se = new Node(n);
			}
		}

		void displayLinesRecursive(Node* root)
		{
			if (root)
			{
				glVertex2f(root->left, root->top);
				glVertex2f(root->right, root->top);

				glVertex2f(root->left, root->top);
				glVertex2f(root->left, root->bottom);

				glVertex2f(root->right, root->top);
				glVertex2f(root->right, root->bottom);

				glVertex2f(root->left, root->bottom);
				glVertex2f(root->right, root->bottom);

				displayLinesRecursive(root->nw);
				displayLinesRecursive(root->ne);
				displayLinesRecursive(root->sw);
				displayLinesRecursive(root->se);
			}
		}

		void displayLines() {
			if (root != 0)
			{
				glBegin(GL_LINES);
				
				
				displayLinesRecursive(root);

				glEnd();
			}
			
		}

		void clear()
		{
			deleteRecursive(root);
			root = 0;
		}

		void deleteRecursive(Node* t)
		{
			if (t == 0) return;
			else {
				deleteRecursive(t->nw);
				deleteRecursive(t->ne);
				deleteRecursive(t->sw);
				deleteRecursive(t->se);
				delete t;
				t = 0;
			}
		}

		~QuadTree() {
			clear();
		}
	};
	QuadTree tree;
	void calcQuadTree() {
		left = right = positions[0]; top = bottom = positions[1];
		for (int i = 1; i < numParticles; i++)
		{
			left = std::min(positions[i * 2], left);
			right = std::max(positions[i * 2], right);
			top = std::max(positions[i * 2 + 1], top);
			bottom = std::min(positions[i * 2 + 1], bottom);
		}

		tree.clear();
		Node n(positions[0], positions[1]);
		n.left = left; n.right = right; n.bottom = bottom; n.top = top;
		tree.insert(n);
		tree.root->left = left; tree.root->right = right; tree.root->top = top; tree.root->bottom = bottom;
		for (int i = 1; i < numParticles; i++)
		{
			Node node(positions[i * 2], positions[i * 2 + 1]);
			tree.insert(node);
		}
			
	}
	// Brute force implementation
	void run() {

		
		for (int i = 0; i < numParticles; i++)
		{
			Vector gravityAcc;
			float dSq;
			for (int j = 0; j < numParticles; j++)
			{
				if (i != j)
				{
					dSq = distSquared(&positions[i*2], &positions[j*2]);
					if (dSq <= 4 * r * r) {
						gravityAcc = Vector(0, 0);
					}
					else gravityAcc = mult(sub(&positions[i*2], &positions[j*2]), dt * G * mass / (dSq * (float)sqrt(dSq + 1)));

					velocities[i * 2] += gravityAcc.x;
					velocities[i * 2 + 1] += gravityAcc.y;
				}
			}
			// Gravity in centre
			float centerPos[2] = { 0.f, 0.f };
			float centerMass = 0.001f;
			dSq = distSquared(&positions[i * 2], centerPos);
			gravityAcc = mult(sub(&positions[i * 2], centerPos), dt * G * mass * centerMass * dSq / (float)sqrt(dSq + 0.0001));
			//velocities[i * 2] += gravityAcc.x;
			//velocities[i * 2 + 1] += gravityAcc.y;

			positions[i * 2] -= velocities[i * 2];
			positions[i * 2 + 1] -= velocities[i * 2 + 1];
		}
		calcQuadTree();
	}

	~NBodySim() {
		delete[] positions;
		delete[] velocities;
	}

	const float* getPositions() { return positions; }
private:
	int numParticles;

	const float bound = 6.f;
	float* positions;
	float* velocities;
	float r; // Radius
	const float mass = 1;
	float G;
	float theta;
	float restitution;
	float initBounds;
	float initVel;
	float dt; // Timestep

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution;

	struct Vector
	{
		float pt[2];
		float& x = pt[0];
		float& y = pt[1];
		Vector(float x, float y) { this->x = x; this->y = y; }
		Vector() { x = 0; y = 0; }
		Vector(const Vector& v) { x = v.x; y = v.y; }
		const Vector& operator=(const Vector& v) { x = v.x; y = v.y; return *this; }
		float magSquared() {
			return x * x + y * y;
		}

		operator float* () { return pt; }
	};

	float dist(float* a, float* b)
	{
		return sqrt(distSquared(a, b));
	}

	float distSquared(float* a, float* b) {
		return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
	}

	float add(float* a, float* b) {
		return (a[0] + b[0], a[1] + b[1]);
	}

	Vector sub(float* a, float* b) {
		return Vector(a[0] - b[0], a[1] - b[1]);
	}

	Vector mult(float* a, float scalar) {
		return Vector(a[0] * scalar, a[1] * scalar);
	}

	Vector normalize(float* a) {
		return mult(a, invSqrt(Vector(a[0], a[1]).magSquared()));
	}

	float dot(float* a, float* b) {
		return a[0] * b[0] + a[1] * b[1];
	}

	// Id Studios
	float invSqrt(float number)
	{
		long i;
		float x2, y;
		const float threehalfs = 1.5F;

		x2 = number * 0.5F;
		y = number;
		i = *(long*)&y;                       // evil floating point bit level hacking
		i = 0x5f3759df - (i >> 1);               // what the fuck? 
		y = *(float*)&i;
		y = y * (threehalfs - (x2 * y * y));   // 1st iteration
	//	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

		return y;
	}
	
};