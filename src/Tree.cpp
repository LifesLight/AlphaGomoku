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
    std::vector<Node*> all_nodes = getAllNodes();
    for (Node* node : all_nodes)
    {
        delete node;
        node = nullptr;
    }
}

void nodeCrawler(std::vector<Node*>& node_vector, Node* node)
{
    node_vector.push_back(node);
    for (Node* child : node->children)
        nodeCrawler(node_vector, child);
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

bool Tree::makeMove(index_t index)
{
    uint8_t x, y;
    Utils::indexToCords(index, x, y);
    return makeMove(x, y);
}

bool Tree::makeMove(uint8_t x, uint8_t y)
{
    index_t move_index;
    Utils::cordsToIndex(move_index, x, y);

    State* current_state = current_node->state;

    // Check if move is legal
    if (!(
        (0 <= x && x < BoardSize) && 
        (0 <= y && y < BoardSize)
        ))
    {
        std::cout << "[Tree][W]: Tried to perform illegal move (Cords out of bounds " << int(x) << "," << int(y) << ")" << std::endl << std::flush;
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

    // We check for if node exists first, if it does the field will be alloctated
    if (!current_state->isCellEmpty(x, y) && chosen_child == nullptr)
    {
        std::cout << "[Tree][W]: Tried to perform illegal move (Allocated field): (" << int(x) << "," << int(y) << ") on:" << std::endl;
        std::cout << current_state->toString() << std::endl << std::flush;
        return false;
    }

    // Node does not have desired child
    if (chosen_child == nullptr)
    {
        // Clear parent children
        for (Node* child : current_node->children)
            deletion_queue.push_back(child);

        // Expand to move index
        chosen_child = current_node->expand(move_index);
        node_count++;

        // Push the new node into the network queue
        network_queue.push_back(chosen_child);
    }
    // Node does have child
    else
    {
        // Delete other children
        for (Node* child : current_node->children)
            if (child != chosen_child)
                deletion_queue.push_back(child);
    }

    current_node = chosen_child;
    return true;
}

Node* Tree::policy()
{
    // Policy loop
    Node* current = current_node;
    while (!current->isTerminal())
    {
        if (current->getUntriedActions().size() > 0)
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

std::list<Node*> Tree::getNetworkQueue()
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
        std::cout << "[Tree][W]: Got non existent parent" << std::endl << std::flush;
    return current_node->parent;
}

void Tree::clean()
{
    for (Node* garbage : deletion_queue)
    {
        // Delete node from queue
        network_queue.remove(garbage);

        // Delete child pointer from children list and switch to frozzen status
        garbage->parent->removeNodeFromChildren(garbage);
        garbage->parent->shrinkNode();

        delete garbage;
    }

    deletion_queue.clear();
}

bool Tree::isTerminal()
{
    return current_node->state->isTerminal();
}