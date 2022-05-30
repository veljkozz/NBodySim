#include "QuadTree.h"

#include <GL/glew.h>

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

