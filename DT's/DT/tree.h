#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

struct Node
{
	bool isLeaf;
	int prediction;
	int featureIdx;
	double threshold;
	Node* left;
	Node* right;
	Node() : isLeaf(false), prediction(-1), featureIdx(-1), threshold(0.0), left(nullptr), right(nullptr){}
};

struct Sample {
	int label;
	std::vector<double> features;
	Sample(std::vector<double> f, int l) : features(f), label(l) {}
};


inline void printSample(const Sample &s) {
	for(int i = 0; i < s.features.size(); i++){
		std::cout << s.features[i] << " ";
	}

	std::cout << "\n" << "label"<< s.label;
}



class DecisionTree {
private:
	int maxDepth;
	Node* root;
	Node* buildTree(std::vector<Sample>& data, int depth);

public:
	DecisionTree(int maxDepth) : maxDepth(maxDepth), root(nullptr) {};
	DecisionTree() :maxDepth(5), root(nullptr) {};
	~DecisionTree();
	void train(std::vector<Sample>& data);
	int predict_single(const Sample& sample) const;
	std::vector<int> predict(const std::vector<Sample>& samples) const;
	double Entropy(const std::vector<Sample>& data) const;
	void deleteTree(Node* node);
	bool isPure(const std::vector<Sample>& data) const;
	int findMajorityClass(const std::vector<Sample>& data) const;
	double findTheBestSplit(std::vector<Sample>& data, int& bestFeature, double& bestThreshold) const;
	double calculateSplitEntropy(const std::map<int, int>& leftCounts, int leftSize,
		const std::map<int, int>& rightCounts, int rightSize) const;
};

