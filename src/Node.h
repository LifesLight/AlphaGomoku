#pragma once
#include "Config.h"
#include "State.h"
#include "Model.h"
#include "NodeData.h"

/*
Node is a singular element in a Tree, it represents a unique board position.
Unlike state, it stores various information about the board position relevant for the algorithm.

It requires a model output for most actions.
This model output is not calculated on creation since we want to batch model calls.
*/

// Data which is not needed after 


class Node
{
public:
    // Permanent Node data
    Node* parent;
    index_t parent_action;
    State* state;
    std::list<Node*> children;

    // Interface to data struct
    // Original evaluation
    float getValueHeadEval();
    // Sum of backprob evals
    float getSummedEvaluation();
    // How often node was visited
    uint32_t getVisits();
    // Get untried actions
    std::deque<index_t>& getUntriedActions();

private:
    // Temporary Node data
    NodeData* temp_data;

public:
    // Deletes temp data
    void shrinkNode();

    // Provide model output
    void setModelOutput(torch::Tensor policy, torch::Tensor value);

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
    float getMeanEvaluation();
    // Get this nodes inital policy eval
    float getNodesPolicyEval();

    // Node is in terminal state
    bool isTerminal();
    // Calls backpropagate with value calculation
    void callBackpropagate();

    // Other
    // Removes action from untried_actions
    void removeFromUntried(index_t action);
    // Has node recieved network data
    bool getNetworkStatus();

    // Still in active MCTS tree
    bool isShrunk();

    // Has no untried actions left, if node is shrunk assume fully expanded
    bool isFullyExpanded();

    // Best child without exploration biases
    Node* absBestChild();
    // Best child for policy
    Node* bestChild();

    // Black is 0, 1 is White, Draw is 2
    uint8_t getResult();

    // Removes a node from children
    void removeNodeFromChildren(Node* node);

    // Statics
    // Create string representation of node parameters
    // Green cell is cell of next move
    static std::string analytics(Node* node, const std::initializer_list<std::string> distributions);
    
    // Convert node to tensor Gamestate representation
    static torch::Tensor nodeToGamestate(Node* node);
    static torch::Tensor nodeToGamestate(Node* node, torch::ScalarType dtype);
    static std::string sliceNodeHistory(Node* node, uint8_t depth);

    // Get moves that lead to this node
    std::deque<index_t> getMoveHistory();

    // Next player color
    bool getNextColor();

private:
    // Get value from policy out tensor
    float getPolicyValue(index_t move);
    // Has network data or not
    bool network_status;
    // Gets called when network data is recieved
    void backpropagate(float eval);
    // Figures out what to do with the valHeads output
    float valueProcessor(float normalized_value);
};