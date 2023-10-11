#include "Environment.h"

Environment::Environment(Model* NNB, Model* NNW)
    : current_state(new State())
{
    std::cout << "Initializing Environment with following trees:" << std::endl;

    // Set model before root initialization
    if (NNB != nullptr)
    {
        Node::setNetwork(NNB);
        neural_network[0] = NNB;
        root_node[0] = new Node();
        current_node[0] = root_node[0];
        is_ai[0] = true;

        std::cout << "  Black: (NW:" << NNB->getName() << ")" << std::endl;
    }

    if (NNW != nullptr)
    {
        Node::setNetwork(NNW);
        neural_network[1] = NNW;
        root_node[1] = new Node();
        current_node[1] = root_node[1];
        is_ai[1] = true;

        std::cout << "  White: (NW:" << NNW->getName() << ")" << std::endl;
    }

    // Init Log Table
    std::memset(log_table, 0, MaxSimulations * sizeof(float));
    for (uint32_t i = 1; i < MaxSimulations; i++)
        log_table[i] = std::log(i);
    Node::setLogTable(log_table);
}

Environment::~Environment()
{
    delete current_state;
    delete root_node[0];
    delete root_node[1];
    delete current_node[0];
    delete current_node[1];
    delete[] log_table;
}

bool Environment::makeMove(uint8_t x, uint8_t y)
{
    bool color = current_state->nextColor();

    // Check if field in bounds
    if (!(0 <= x && x < BoardSize)) return false;
    if (!(0 <= y && y < BoardSize)) return false;
    if (!current_state->isCellEmpty(x, y)) return false;

    std::cout << "[Env]: Making move [" << int(x) << "," << int(y) << "] for ";
    if (color)
        std::cout << "White" << std::endl;
    else
        std::cout << "Black" << std::endl;

    uint16_t move_index;
    Utils::cordsToIndex(move_index, x ,y);
    
    // Update current state
    current_state->makeMove(move_index);

    if (is_ai[color])
    {
        std::cout << "[Env]: Updating ";
        if (color)
            std::cout << "White Tree" << std::endl;
        else
            std::cout << "Black Tree" << std::endl;

        Node* working_node = current_node[color];
        Node* chosen_child = nullptr;

        // Get matching child
        for (Node* child : working_node->children)
            if (child->parent_action == move_index)
            {
                chosen_child = child;
                break;
            }

        // Node does not have desired child
        if (chosen_child == nullptr)
        {
            std::cout << " ^[W]: Manually expanding Tree (expected existing child)" << std::endl;

            // Clear parent children
            for (Node* child : working_node->children)
                delete child;
            working_node->children.clear();

            chosen_child = new Node(new State(current_state), working_node, move_index);
            working_node->children.push_back(chosen_child); 
        }
        else
        {
            // Constrain Parent
            working_node->constrain(chosen_child);
        }

        // Set the updated node to current
        current_node[color] = chosen_child;
    }

    // Check if opposite tree exists if yes update
    if (is_ai[!color])
    {
        std::cout << "[Env]: Updating ";
        if (!color)
            std::cout << "White Tree" << std::endl;
        else
            std::cout << "Black Tree" << std::endl;
            
        Node* opposing_node = current_node[!color];
        Node* target_node = nullptr;
        
        // Find opposing node that matches this
        for (Node* child : opposing_node->children)
        {
            if (child->parent_action == move_index)
            {
                target_node = child;
                break;
            }
        }

        // If opposing tree doesnt have node
        if (target_node == nullptr)
        {
            std::cout << " ^[I]: Manually expanding Tree" << std::endl;

            // Clear parent children
            for (Node* child : opposing_node->children)
                delete child;
            opposing_node->children.clear();

            target_node = new Node(new State(current_state), opposing_node, move_index);
            opposing_node->children.push_back(target_node); 
        }
        else
        {
            opposing_node->constrain(target_node);
        }

        // Set the updated node to current
        current_node[!color] = target_node;
    }

    return true;
}

uint16_t Environment::calculateNextMove(uint32_t simulations)
{
    bool color = current_state->nextColor();

    if (!is_ai[color])
    {
        std::cout << "[Environment]: Error, tried to calculateNextMove for non AI player" << std::endl;
        return -1;
    }

    Node* working_node = current_node[color];
    Node::setHeadNode(working_node);
    Node::setNetwork(neural_network[color]);

    for (int i = 0; i < simulations; i++)
    {
        working_node->simulationStep();
    }

    // Get best child
    Node* best_child = working_node->absBestChild();
    return best_child->parent_action;
}

std::string Environment::toString()
{
    return current_state->toString();
}

std::string Environment::toString(uint8_t depth)
{
    bool color = current_state->nextColor();
    Gamestate current_gamestate(current_node[color]);
    return current_gamestate.sliceToString(depth);
}


bool Environment::makeMove(uint16_t index)
{
    uint8_t x, y;
    Utils::indexToCords(index, x, y);
    return makeMove(x, y);
}

bool Environment::isFinished()
{
    return current_state->isTerminal();
}

Node* Environment::getNode()
{
    bool color = current_state->nextColor();
    return current_node[color];
}


Node* Environment::getNode(bool color)
{
    return current_node[color];
}
