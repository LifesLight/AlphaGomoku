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
    std::vector<std::tuple<uint16_t, float>> untried_actions;

    // Network stuff
    float value;
    // For get prior propability
    std::map<uint16_t, float> policy_evaluations;

    void addNetworkOutput(std::tuple<torch::Tensor, torch::Tensor> model_output);
    float getPriorPropability();
    bool hasNetworkOut();

    // Constructors
    Node(State* state, Node* parent, uint16_t parent_action);
    Node(State* state);
    Node();
    ~Node();


    Node* bestChild();
    // Best child without exploration biases
    Node* absBestChild();
    // Best child within confidence bound
    Node* absBestChild(float);

    Node* simulationStep();
    float meanEvaluation();
    void backpropagate(float value, Node* head_node);

    // Config
    static void setLogTable(float* log_table);
    static std::string analytics(Node* node, const std::initializer_list<std::string> distributions);

private:
    static float* logTable;
    bool has_network_output;
    Node* expand();
};