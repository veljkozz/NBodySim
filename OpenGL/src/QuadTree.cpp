#include "QuadTree.h"

#include <GL/glew.h>
#include "NBodySeq.h"
#include "utils.h"

void QuadTree::insert(Node& n)
{
	if (root)
		insertRecursive(root, n);
	else root = new Node(n);
}

void QuadTree::insertRecursive(Node* t, Node& n)
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


void QuadTree::insertRecursiveChoice(Node* t, Node n)
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


void QuadTree::updateParentChild(Node* parent, Node* child)
{
	parent->mass += child->mass;
	parent->pos[0] += child->mass * child->pos[0];
	parent->pos[1] += child->mass * child->pos[1];
}

void QuadTree::updateMass(Node* t)
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

void QuadTree::displayLinesRecursive(Node* root)
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


void QuadTree::displayLines() {
	if (root != 0)
	{
		glBegin(GL_LINES);

		displayLinesRecursive(root);

		glEnd();
	}

}


void QuadTree::clear()
{
	deleteRecursive(root);
	root = 0;
}

void QuadTree::forceCalculations(int i)
{
	numCalcs = 0;
	calcForceRecursive(root, i);
}


void QuadTree::deleteRecursive(Node* t)
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



QuadTree::~QuadTree() {
	clear();
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
	if (t->leaf)
	{
		if(t->id != i) calcForce(t, i);
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


int QuadTree::getNumNodes()
{
	if (numNodes == 0)
		getNumNodesRecursive(root);
	return numNodes;
}
void QuadTree::getNumNodesRecursive(Node* t)
{
	numNodes++;
	if (t->nw) getNumNodesRecursive(t->nw);
	if (t->ne) getNumNodesRecursive(t->ne);
	if (t->sw) getNumNodesRecursive(t->sw);
	if (t->se) getNumNodesRecursive(t->se);
}
