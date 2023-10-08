#pragma once
#include "Config.h"
#include "State.h"

class Node
{
public:
    Node* parent;
    uint16_t parent_action;
    State* state;
    uint32_t visits;
    uint32_t results[3];
    std::list<Node*> children;
    std::vector<std::tuple<uint16_t, float>> untried_actions;

    // Network stuff
    float value;
    float prior_propability;

    Node();
    Node(State*);
    Node(State*, Node*, uint16_t);
    Node(Node*);
    ~Node();

    void rollout();
    Node* bestChild();
    
    // Best child without exploration biases
    Node* absBestChild();
    // Best child within confidence bound
    Node* absBestChild(float);

    Node* policy();
    int32_t qDelta(bool);

private:
    Node* expand();
    void backpropagate(uint8_t);
};