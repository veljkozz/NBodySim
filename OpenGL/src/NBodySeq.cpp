#include "NBodySeq.h"
#include <GL/glew.h>
#include <chrono>
#include <iostream> 
#include "Params.h"
#include "utils.h"

NBodySeq::NBodySeq(int numParticles) : numParticles(numParticles)
{
	positions = new float[2 * numParticles];
	velocities = new float[2 * numParticles];

	distribution = std::uniform_real_distribution<double>(lowbound, bound);
	std::uniform_real_distribution<double> distributionPI(0, 2 * PI);
	for (int i = 0; i < numParticles; i++)
	{
		float angle = distributionPI(generator); // (0, TWO_PI);
		float dist = distribution(generator);
		float mag = 0;// dist * 0.002;

		positions[i * 2] = dist * cos(angle);
		positions[i * 2 + 1] = dist * sin(angle);
		velocities[i * 2] = mag * cos(angle + PI / 2);
		velocities[i * 2 + 1] = mag * sin(angle - PI / 2);
	}
}


void NBodySeq::QuadTree::insert(Node & n)
{
	if (root)
		insertRecursive(root, n);
	else root = new Node(n);
}

void NBodySeq::QuadTree::insertRecursive(Node* t, Node& n)
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
	updateMass(t);
}


void NBodySeq::QuadTree::insertRecursiveChoice(Node* t, Node n)
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


void NBodySeq::QuadTree::updateParentChild(Node* parent, Node* child)
{
	parent->mass += child->mass;
	parent->pos[0] += child->mass * child->pos[0];
	parent->pos[1] += child->mass * child->pos[1];
}

void NBodySeq::QuadTree::updateMass(Node* t)
{
	t->mass = 0;
	t->pos[0] = t->pos[1] = 0;
	if (t->nw) updateParentChild(t, t->nw);
	if (t->ne) updateParentChild(t, t->ne);
	if (t->sw) updateParentChild(t, t->sw);
	if (t->se) updateParentChild(t, t->se);
	t->pos[0] /= t->mass;
	t->pos[1] /= t->mass;
}

void NBodySeq::QuadTree::displayLinesRecursive(Node* root)
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


void NBodySeq::QuadTree::displayLines() {
	if (root != 0)
	{
		glBegin(GL_LINES);

		displayLinesRecursive(root);

		glEnd();
	}

}


void NBodySeq::QuadTree::clear()
{
	deleteRecursive(root);
	root = 0;
}


void NBodySeq::QuadTree::deleteRecursive(Node* t)
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

NBodySeq::QuadTree::~QuadTree() {
	clear();
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
	Node n(positions[0], positions[1], 0);
	n.left = left; n.right = right; n.bottom = bottom; n.top = top;
	tree.insert(n);
	tree.root->left = left; tree.root->right = right; tree.root->top = top; tree.root->bottom = bottom;
	for (int i = 1; i < numParticles; i++)
	{
		Node node(positions[i * 2], positions[i * 2 + 1], i);
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
	numCalcs = 0;

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
		calcForceRecursive(tree.root, i);

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

void NBodySeq::calcForce(Node* t, int i)
{
	float dSq = distSquared(&positions[i * 2], &positions[t->id * 2]);
	Vector acc;
	if (dSq <= 4 * r * r) {
		acc = Vector(0, 0);
	}
	//  PAZI STA JE MASA OVDE
	else acc = mult(sub(&positions[i * 2], &positions[t->id * 2]), dt * G * t->mass / (dSq * sqrt(dSq) + 2));

	velocities[i * 2] -= acc.x;
	velocities[i * 2 + 1] -= acc.y;
	numCalcs++;
}

void NBodySeq::calcForceRecursive(Node* t, int i)
{
	if (t == 0) return;
	if (t->leaf && t->id != i)
	{
		calcForce(t, i);
	}
	else {
		float s = t->right - t->left;
		float d = dist(&positions[i], t->pos);
		// if s/d < theta then treat this as a single body
		if (s / d < theta)
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