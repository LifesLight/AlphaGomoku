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
    Model* neural_network;
    float value;
    float prior_propability;

    
    Node(State*, Node*, uint16_t, Model*);
    Node(State*, Model*);
    Node(Node*, Model*);
    Node(Model*);

    ~Node();

    void rollout();
    Node* bestChild();
    
    // Best child without exploration biases
    Node* absBestChild();
    // Best child within confidence bound
    Node* absBestChild(float);

    Node* policy();
    float meanEvaluation(bool);

private:
    Node* expand();
    void backpropagate(float);
};