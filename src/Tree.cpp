#include "Tree.h"

Tree::Tree()
{
    root_node = new Node();
    network_queue.push_back(root_node);
    current_node = root_node;
}

Tree::~Tree()
{
    // Not correct?
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

        // Remove untried from parent node
        current_node->removeFromUntried(move_index);

        // Create new node
        State* updated_state = new State(current_state);
        updated_state->makeMove(move_index);
        chosen_child = new Node(updated_state, current_node, move_index);
        current_node->children.push_back(chosen_child);

        // Push the new node into the network queue
        network_queue.push_back(chosen_child);
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

Node* Tree::simulationStep()
{
    // Policy loop
    Node* current = current_node;
    while (!current->isTerminal())
    {
        if (current->untried_actions.size() > 0)
        {
            Node* new_node = current->expand();
            network_queue.push_back(new_node);
            return new_node;
        }
        else
        {
            current = current->bestChild();
        }
    }
    return current;
}

std::vector<Node*> Tree::getNetworkQueue()
{
    return network_queue;
}

bool Tree::clearNetworkQueue()
{
    std::vector<Node*> unsuccessfull;
    for (Node* node : network_queue)
        if (!node->getNetworkStatus())
            unsuccessfull.push_back(node);
    
    network_queue.clear();

    // All queued nodes are sucessfully initialized
    if (unsuccessfull.size() == 0)
        return true;

    // Some nodes are still not initialized
    for (Node* node : unsuccessfull)
        network_queue.push_back(node);
    return false;
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
    {
        // Remove node from parent children list
        garbage->parent->children.remove(garbage);
        delete garbage;
    }
        
    deletion_queue.clear();
}

bool Tree::isTerminal()
{
    return current_node->state->isTerminal();
}