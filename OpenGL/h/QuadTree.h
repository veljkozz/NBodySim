#pragma once

class NBodySeq;

class QuadTree {
public:

	QuadTree(NBodySeq* simulation) : sim(simulation) {}

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
	
	void insert(Node& n);
	
	
	void displayLines();
	void clear();
	
	void forceCalculations(int i);

	int getNumCalcs() { return numCalcs; }
	
	
	~QuadTree();

private:
	Node* root;
	NBodySeq* sim;
	int numCalcs;

	void insertRecursive(Node* t, Node& n);
	void insertRecursiveChoice(Node* t, Node n);
	void updateParentChild(Node* parent, Node* child);
	void updateMass(Node* t);

	void displayLinesRecursive(Node* root);
	void deleteRecursive(Node* t);
	void calcForce(Node* t, int i);
	void calcForceRecursive(Node* t, int i);
};