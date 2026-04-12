#include "tree.h"
#include <cmath>
#include <map>
#include <algorithm>

DecisionTree::~DecisionTree() {
    deleteTree(root);
}

void DecisionTree::deleteTree(Node* node) {
    if (node) {
        deleteTree(node->left);
        deleteTree(node->right);
        delete node;
    }
}

double DecisionTree::Entropy(const std::vector<Sample>& data) const {
    if (data.empty()) return 0.0;

    std::map<int, int> counts;
    for (const auto& s : data) {
        counts[s.label]++;
    }

    double entropy = 0.0;
    double total = static_cast<double>(data.size());

    for (auto const& [label, count] : counts) {
        double p = count / total;
        entropy -= p * std::log2(p);
    }
    return entropy;
}

bool DecisionTree::isPure(const std::vector<Sample>& data) const {
    if (data.empty()) return true;
    int firstLabel = data[0].label;
    for (const auto& s : data) {
        if (s.label != firstLabel) return false;
    }
    return true;
}

int DecisionTree::findMajorityClass(const std::vector<Sample>& data) const {
    std::map<int, int> counts;
    for (const auto& s : data) counts[s.label]++;

    int majorityClass = -1;
    int maxCount = -1;
    for (auto const& [label, count] : counts) {
        if (count > maxCount) {
            maxCount = count;
            majorityClass = label;
        }
    }
    return majorityClass;
}

double DecisionTree::findTheBestSplit(std::vector<Sample>& data, int& bestFeature, double& bestThreshold) const {
    double maxGain = -1.0;
    double parentEntropy = Entropy(data);
    int n_f = data[0].features.size();
    int n_s = data.size();

    for (int f = 0; f < n_f; ++f) {
        std::sort(data.begin(), data.end(), [f](const Sample& a, const Sample& b) {
            return a.features[f] < b.features[f];
            });

        std::map<int, int> leftCounts, rightCounts;
        for (const auto& s : data) rightCounts[s.label]++;

        for (int i = 0; i < n_s - 1; ++i) {
            int label = data[i].label;
            leftCounts[label]++;
            rightCounts[label]--;

            if (data[i].features[f] < data[i + 1].features[f]) {
                double threshold = (data[i].features[f] + data[i + 1].features[f]) / 2.0;

                double gain = parentEntropy - calculateSplitEntropy(leftCounts, i + 1, rightCounts, n_s - i - 1);

                if (gain > maxGain) {
                    maxGain = gain;
                    bestFeature = f;
                    bestThreshold = threshold;
                }
            }
        }
    }
    return maxGain;
}

Node* DecisionTree::buildTree(std::vector<Sample>& data, int depth) {
    if (data.empty()) return nullptr;

    Node* node = new Node();
    int bestFeature = -1;
    double bestThreshold = 0.0;
    double gain = findTheBestSplit(data, bestFeature, bestThreshold);

    if (isPure(data) || depth >= maxDepth || gain <= 0) {
        node->isLeaf = true;
        node->prediction = findMajorityClass(data);
        return node;
    }

    node->featureIdx = bestFeature;
    node->threshold = bestThreshold;

    std::vector<Sample> leftData, rightData;
    for (const auto& s : data) {
        if (s.features[bestFeature] <= bestThreshold) leftData.push_back(s);
        else rightData.push_back(s);
    }

    node->left = buildTree(leftData, depth + 1);
    node->right = buildTree(rightData, depth + 1);

    return node;
}

double DecisionTree::calculateSplitEntropy(const std::map<int, int>& leftCounts, int leftSize,
    const std::map<int, int>& rightCounts, int rightSize) const {
    auto computeEntropy = [](const std::map<int, int>& counts, int total) {
        if (total == 0) return 0.0;
        double entropy = 0.0;
        for (auto const& [label, count] : counts) {
            if (count == 0) continue;
            double p = static_cast<double>(count) / total;
            entropy -= p * std::log2(p);
        }
        return entropy;
        };

    double totalSamples = leftSize + rightSize;
    double weightLeft = leftSize / totalSamples;
    double weightRight = rightSize / totalSamples;

    return (weightLeft * computeEntropy(leftCounts, leftSize)) +
        (weightRight * computeEntropy(rightCounts, rightSize));
}

void DecisionTree::train(std::vector<Sample>& data) {
    root = buildTree(data, 0);
}

int DecisionTree::predict_single(const Sample& sample) const {
    Node* curr = root;
    while (curr && !curr->isLeaf) {
        if (sample.features[curr->featureIdx] <= curr->threshold) {
            curr = curr->left;
        }
        else {
            curr = curr->right;
        }
    }
    return curr ? curr->prediction : -1;
}

std::vector<int> DecisionTree::predict(const std::vector<Sample>& samples) const {
    std::vector<int> predictions;
    for (const auto& s : samples) {
        predictions.push_back(predict_single(s));
    }
    return predictions;
}