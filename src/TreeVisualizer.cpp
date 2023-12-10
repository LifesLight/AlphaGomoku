/**
 * Copyright (c) Alexander Kurtz 2023
*/


#include "TreeVisualizer.h"

void TreeVisualizer::generateGraphvizCode(Node* root, std::ostream& out) {
    out << "digraph node_tree {" << std::endl;
    out << "  rankdir=TB;" << std::endl;
    out << "  ranksep=25.0;" << std::endl;
    out << "  node [shape=box];" << std::endl;

    if (root) {
        int nextId = 0;
        std::unordered_map<Node*, int> nodeIds;

        TreeVisualizer::traverseAndGenerateCode(root, out, nextId, nodeIds);
    }

    out << "}" << std::endl;
}

void TreeVisualizer::traverseAndGenerateCode(Node* node, std::ostream& out, int& nextId, std::unordered_map<Node*, int>& nodeIds) {
    // Assign an ID to the node if it doesn't have one
    if (nodeIds.count(node) == 0) {
        nodeIds[node] = nextId++;
    }

    // Write the node to the output
    std::string color = (node->getProcessedEval() == 1.0f) ? "green" : ((node->getProcessedEval() == -1.0f) ? "red" : "black");
    // If processed eval is 1 print 1-Proc, -1 is -1-Proc
    if (node->getProcessedEval() == 1.0f) {
        out << " " << nodeIds[node] << " [label=\"" << node->getProcessedEval() << "-Proc\", color=\"" << color << "\"];" << std::endl;
    } else if (node->getProcessedEval() == -1.0f) {
        out << " " << nodeIds[node] << " [label=\"" << node->getProcessedEval() << "-Proc\", color=\"" << color << "\"];" << std::endl;
    } else {
        out << " " << nodeIds[node] << " [label=\"" << node->getProcessedEval() << "\", color=\"" << color << "\"];" << std::endl;
    }

    // Visit each child
    for (Node* child : node->children) {
        // Assign an ID to the child if it doesn't have one
        if (nodeIds.count(child) == 0) {
            nodeIds[child] = nextId++;
        }

        // Write the edge to the output
        out << " " << nodeIds[node] << " -> " << nodeIds[child] << ";" << std::endl;

        // Recurse on the child
        traverseAndGenerateCode(child, out, nextId, nodeIds);
    }
}
