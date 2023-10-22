#pragma once
#include "Config.h"
#include "State.h"
#include "Model.h"
#include "Gamestate.h"

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
    float prior_propability;
    bool is_initialized;

    void runNetwork(Model* neural_net);

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

    void simulationStep(Model* neural_network);
    float meanEvaluation();

    // Config
    static void setLogTable(float* log_table);
    static std::string analytics(Node* node, const std::initializer_list<std::string> distributions);

private:
    static float* logTable;

    Node* expand(Model* neural_network);
    void backpropagate(float value, Node* head_node);
};