#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
*/


#include "Config.h"
#include "Node.h"

class TreeVisualizer
{
public:
    static void generateGraphvizCode(Node* root, std::ostream& out);

private:
    static void traverseAndGenerateCode(Node* node, std::ostream& out, int& nextId, std::unordered_map<Node*, int>& nodeIds);

    // Prevent instantiation
    TreeVisualizer() = delete;
};
