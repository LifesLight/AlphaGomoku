#pragma once
#include "Config.h"
#include "State.h"
#include "Model.h"

/*
Node is a singular element in a Tree, it represents a unique board position.
Unlike state, it stores various information about the board position relevant for the algorithm.

It requires a model output for most actions.
This model output is not calculated on creation since we want to batch model calls.
*/

class Node
{
public:
    Node* parent;
    index_t parent_action;
    State* state;
    uint32_t visits;
    std::list<Node*> children;
    std::deque<index_t> untried_actions;
    float evaluation;
    float summed_evaluation;
    float policy_evaluations[BoardSize * BoardSize];

    // Neural Net
    // Get inital policy evaluation for this node
    float getPolicyValue();
    // Provide model output
    void setModelOutput(std::tuple<torch::Tensor, torch::Tensor> input);

    // Constructors
    Node(State* state, Node* parent, index_t parent_action);
    Node(State* state);
    Node();
    ~Node();

    // Algorithm
    // Auto expand by policy values
    Node* expand();
    // Manual expand with move_index
    Node* expand(index_t move_index);
    // Evaluation of node according to MCTS
    float meanEvaluation();
    // Node is in terminal state
    bool isTerminal();
    // Calls backpropagate with value calculation
    void callBackpropagate();

    // Other
    // Removes action from untried_actions
    void removeFromUntried(index_t action);
    // Has node recieved network data
    bool getNetworkStatus();

    // Best child without exploration biases
    Node* absBestChild();
    // Best child for policy
    Node* bestChild();

    // Black is 0, 1 is White, Draw is 2
    uint8_t getResult();

    // Statics
    // Create string representation of node parameters
    // Green cell is cell of next move
    static std::string analytics(Node* node, const std::initializer_list<std::string> distributions);
    
    // Convert node to tensor Gamestate representation
    static torch::Tensor nodeToGamestate(Node* node);
    static std::string sliceNodeHistory(Node* node, uint8_t depth);

    // Get moves that lead to this node
    std::deque<index_t> getMoveHistory();

    // Next player color
    bool getNextColor();

private:
    // Has network data or not
    bool network_status;
    // Gets called when network data is recieved
    void backpropagate(float eval);
    // Figures out what to do with the valHeads output
    float valueProcessor(float normalized_value);
};