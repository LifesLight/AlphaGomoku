#pragma once
#include "Includes.h"
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

    // Constructors
    Node(State*, Node*, uint16_t);
    Node(State*);
    Node(Node*);
    Node();
    ~Node();


    Node* bestChild();
    // Best child without exploration biases
    Node* absBestChild();
    // Best child within confidence bound
    Node* absBestChild(float);

    void simulationStep();
    float meanEvaluation(bool);

    // Config
    static void setNetwork(Model* neural_net);
    static void setLogTable(float* log_table);
    static void setHeadNode(Node* head);

    // Constrain children to one Node
    void constrain(Node* valid);

private:
    static Model* neural_network;
    static float* logTable;
    static Node* head_node;

    Node* expand();
    void backpropagate(float);
};