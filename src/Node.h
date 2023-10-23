#pragma once
#include "Config.h"
#include "State.h"
#include "Model.h"

class Node
{
public:
    Node* parent;
    uint16_t parent_action;
    State* state;
    uint32_t visits;
    float summed_evaluation;
    std::list<Node*> children;
    std::vector<uint16_t> untried_actions;
    float policy_evaluations[BoardSize * BoardSize];

    // Neural Net
    float evaluation;
    float getPolicyValue();

    // Constructors
    Node(State* state, Node* parent, uint16_t parent_action);
    Node(State* state);
    Node();
    ~Node();

    // Algorithm
    Node* expand();
    float meanEvaluation();
    void backpropagate(float value, Node* head_node);
    Node* bestChild();

    // Best child without exploration biases
    Node* absBestChild();

    // Debug
    static std::string analytics(Node* node, const std::initializer_list<std::string> distributions);
};