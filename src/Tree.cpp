#include "Tree.h"

Tree::Tree(Model* neural_network)
    : neural_net(neural_network)
{
    root_node = new Node();
    root_node->runNetwork(neural_net);
    current_node = root_node;
}

Tree::~Tree()
{
    delete root_node;
}

bool Tree::makeMove(uint16_t index)
{
    uint8_t x, y;
    Utils::indexToCords(index, x, y);
    return makeMove(x, y);
}

bool Tree::makeMove(uint8_t x, uint8_t y)
{
    uint16_t move_index;
    Utils::cordsToIndex(move_index, x, y);

    State* current_state = current_node->state;

    // Check if move is legal
    if (!(
        (0 <= x && x < BoardSize) && 
        (0 <= y && y < BoardSize) && 
        current_state->isCellEmpty(x, y)
        ))
    {
        std::cout << "[Tree][E]: Tried to perform illegal move" << std::endl;
        return false;
    }

    Node* chosen_child = nullptr;

    // Get matching child
    for (Node* child : current_node->children)
        if (child->parent_action == move_index)
        {
            chosen_child = child;
            break;
        }

    // Node does not have desired child
    if (chosen_child == nullptr)
    {
        // Clear parent children
        for (Node* child : current_node->children)
            deletion_queue.push_back(child);

        // Create new node
        State* updated_state = new State(current_state);
        updated_state->makeMove(move_index);
        chosen_child = new Node(new State(updated_state), current_node, move_index);
        chosen_child->runNetwork(neural_net);
        current_node->children.push_back(chosen_child);
    }
    // Node does have child
    else
    {
        for (Node* child : current_node->children)
            if (child != chosen_child)
                deletion_queue.push_back(child);
    }

    current_node = chosen_child;
    return true;
}

uint16_t Tree::computeMove(uint32_t simulations)
{
    for (uint32_t i = 0; i < simulations; i++)
        current_node->simulationStep(neural_net);
    Node* best_child = current_node->absBestChild();
    return best_child->parent_action;
}

Node* Tree::getCurrentNode()
{
    return current_node;
}

Node* Tree::getParentNode()
{
    if (current_node->parent == nullptr)
        std::cout << "[Tree][W]: Got non existent parent" << std::endl;
    return current_node->parent;
}

void Tree::clean()
{
    for (Node* garbage : deletion_queue)
        delete garbage;
    deletion_queue.clear();
}

bool Tree::isTerminal()
{
    return current_node->state->isTerminal();
}