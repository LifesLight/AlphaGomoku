#include "Tree.h"

Tree::Tree()
    : node_count(0)
{
    root_node = new Node();
    network_queue.push_back(root_node);
    current_node = root_node;
}

Tree::~Tree()
{
    // Will recursively delete all nodes
    delete root_node;
}

void nodeCrawler(std::vector<Node*>& node_vector, Node* node)
{
    node_vector.push_back(node);
    for (Node* child : node->children)
        nodeCrawler(node_vector, child);
}

void Tree::updateCurrentNode(index_t action)
{
    index_t x, y;
    Utils::indexToCords(action, x, y);

    Node* chosen_child = nullptr;
    State* current_state = current_node->state;

    // Put all other children in deletion queue and try to find matching child
    for (Node* child : current_node->children)
        if (child->parent_action != action)
            deletion_queue.push_back(child);
        else
            chosen_child = child;

    // We check for if node exists first, if it does the field will be alloctated
    if (!current_state->isCellEmpty(x, y) && chosen_child == nullptr)
    {
        std::cout << "[Tree][E]: Error in update current child, missmatch between empty cell and child: (" << int(x) << "," << int(y) << ") on:" << std::endl;
        ForcePrintln(current_state->toString());
        return;
    }

    // Node does not have desired child
    if (chosen_child == nullptr)
    {
        // Expand to move index
        chosen_child = current_node->expand(action);
        node_count++;

        // Push the new node into the network queue
        network_queue.push_back(chosen_child);
    }

    if (current_node->parent)
        current_node->parent->shrinkNode();
    current_node = chosen_child;
}

std::vector<Node*> Tree::getAllNodes()
{
    std::vector<Node*> tree_nodes;
    Node* root = getRootNode();
    nodeCrawler(tree_nodes, root);
    return tree_nodes;
}

Node* Tree::getRootNode()
{
    return root_node;
}

int Tree::getNodeCount()
{
    return node_count;
}

void Tree::makeMove(index_t index)
{
    uint8_t x, y;
    Utils::indexToCords(index, x, y);
    makeMove(x, y);
}

void Tree::makeMove(uint8_t x, uint8_t y)
{
    index_t action;
    Utils::cordsToIndex(action, x, y);

    State* current_state = current_node->state;

    // Check if move is legal
    if (!(
        (0 <= x && x < BoardSize) && 
        (0 <= y && y < BoardSize)
        ))
    {
        ForcePrintln("[Tree][W]: Tried to perform illegal move (Cords out of bounds " << int(x) << "," << int(y) << ")");
        return;
    }

    updateCurrentNode(action);
}

Node* Tree::policy()
{
    // Policy loop
    Node* current = current_node;
    while (!current->isTerminal())
    {
        if (!current->isFullyExpanded())
        {
            Node* new_node = current->expand();
            node_count++;
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

void Tree::forceClearNetworkQueue()
{
    network_queue.clear();
}

Node* Tree::getCurrentNode()
{
    return current_node;
}

Node* Tree::getParentNode()
{
    if (current_node->parent == nullptr)
        ForcePrintln("[Tree][W]: Got non existent parent");
    return current_node->parent;
}

void Tree::clean()
{
    for (Node* garbage : deletion_queue)
    {
        // Delete node from queue
        Utils::eraseFromVector(network_queue, garbage);

        // Delete child pointer from children list and switch to frozen status
        garbage->parent->removeNodeFromChildren(garbage);
        delete garbage;
    }

    deletion_queue.clear();
}

bool Tree::isTerminal()
{
    return current_node->state->isTerminal();
}