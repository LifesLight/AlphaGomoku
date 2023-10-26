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
    std::list<Node*> children;
    std::vector<uint16_t> untried_actions;
    bool network_status;
    float evaluation;
    float summed_evaluation;
    float policy_evaluations[BoardSize * BoardSize];

    // Neural Net
    float getPolicyValue();
    bool getNetworkStatus();
    void setModelOutput(std::tuple<torch::Tensor, torch::Tensor> input);

    // Converters
    static torch::Tensor nodeToGamestate(Node* node);
    static std::string sliceNodeHistory(Node* node, uint8_t depth);

    // Constructors
    Node(State* state, Node* parent, uint16_t parent_action);
    Node(State* state);
    Node();
    ~Node();

    // Algorithm
    Node* expand();
    float meanEvaluation();
    void backpropagate(float value);
    Node* bestChild();
    bool isTerminal();

    // Other
    void removeFromUntried(uint16_t action);

    // Best child without exploration biases
    Node* absBestChild();

    // Debug
    static std::string analytics(Node* node, const std::initializer_list<std::string> distributions);
};