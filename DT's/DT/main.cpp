#include "tree.h"
#include <iostream>
#include <vector>
#include <map>

class DatasetLoader {
public:
    std::map<std::string, int> labelMap;
    int nextLabelId = 0;

    std::vector<Sample> loadGeneralCSV(const std::string& path, bool skipHeader = true) {
        std::vector<Sample> dataset;
        std::ifstream file(path);

        if (!file.is_open()) {
            throw std::runtime_error("Fatal: Could not open file at " + path);
        }

        std::string line;
        if (skipHeader) std::getline(file, line);

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> tokens;

     
            while (std::getline(ss, cell, ',')) {
                tokens.push_back(cell);
            }

            if (tokens.size() < 2) continue;

            
            std::vector<double> features;
            for (size_t i = 0; i < tokens.size() - 1; ++i) {
                features.push_back(std::stod(tokens[i]));
            }

            
            std::string rawLabel = tokens.back();

            
            if (labelMap.find(rawLabel) == labelMap.end()) {
                labelMap[rawLabel] = nextLabelId++;
            }

            dataset.emplace_back(features, labelMap[rawLabel]);
        }

        return dataset;
    }

    void printStats() {
        std::cout << "\n--- Dataset Statistics ---" << std::endl;
        for (auto const& [name, id] : labelMap) {
            std::cout << "Class: " << name << " -> Assigned ID: " << id << std::endl;
        }
    }
};

#include "tree.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

int main() {
    DatasetLoader loader;

    try {
        auto fullData = loader.loadGeneralCSV("iris.csv", true);

        if (fullData.empty()) {
            std::cout << "Dataset is empty. Check your CSV path!" << std::endl;
            return 1;
        }

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(fullData.begin(), fullData.end(), g);

        size_t trainSize = static_cast<size_t>(fullData.size() * 0.8);
        std::vector<Sample> trainData(fullData.begin(), fullData.begin() + trainSize);
        std::vector<Sample> testData(fullData.begin() + trainSize, fullData.end());

        std::cout << "Loaded Iris Dataset." << std::endl;
        loader.printStats();
        std::cout << "Training on: " << trainData.size() << " samples" << std::endl;
        std::cout << "Testing on: " << testData.size() << " samples" << std::endl;

        DecisionTree irisTree(4);
        irisTree.train(trainData);


        std::vector<int> results = irisTree.predict(testData);

        int correct = 0;
        for (size_t i = 0; i < testData.size(); ++i) {
            if (results[i] == testData[i].label) {
                correct++;
            }
        }

        double accuracy = (static_cast<double>(correct) / testData.size()) * 100.0;
        std::cout << "\n>>> Final Iris Accuracy: " << accuracy << "% <<<" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}