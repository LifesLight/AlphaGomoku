#include "Environment.h"

Environment::Environment(Model* NNB, Model* NNW)
    :   current_state(new State())
{
    std::cout << "[Env]: Initializing with following trees:" << std::endl;

    // Set model before root initialization
    if (NNB != nullptr)
    {
        neural_network[0] = NNB;
        root_node[0] = new Node();
        root_node[0]->runNetwork(NNB);
        current_node[0] = root_node[0];
        is_ai[0] = true;

        std::cout << "  Black: (NW:" << NNB->getName() << ")" << std::endl;
    }

    if (NNW != nullptr)
    {
        neural_network[1] = NNW;
        root_node[1] = new Node();
        root_node[1]->runNetwork(NNW);
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

            // Create new node
            chosen_child = new Node(new State(current_state), working_node, move_index);
            chosen_child->runNetwork(neural_network[color]);
            working_node->children.push_back(chosen_child); 
        }
        else
        {
            // Constrain Parent
            //working_node->constrain(chosen_child);
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

            // Create new node
            target_node = new Node(new State(current_state), opposing_node, move_index);
            target_node->runNetwork(neural_network[!color]);
            opposing_node->children.push_back(target_node); 
        }
        else
        {
            //opposing_node->constrain(target_node);
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
        std::cout << "[Env]: Error, tried to calculateNextMove for non AI player" << std::endl;
        return -1;
    }

    std::cout << "[Env]: Calculating best move for ";
    
    if (color)
        std::cout << "White" << std::endl;
    else
        std::cout << "Black" << std::endl;

    Node* working_node = current_node[color];
    Model* neural_net = neural_network[color];

    for (int i = 0; i < simulations; i++)
    {
        working_node->simulationStep(neural_net, working_node);
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
    return getNode(color);
}


Node* Environment::getNode(bool color)
{
    if (is_ai[color])
        return current_node[color];

    std::cout << "[Env][W]: Tried to get node of non existent tree" << std::endl;
    return nullptr;
}

bool Environment::getNextColor()
{
    return current_state->nextColor();
}

std::string Environment::nodeAnalytics(Node* node)
{
    bool color = node->state->nextColor();

    std::ostringstream output;
    output << std::endl;

    // Static window
    uint16_t window_width = 40;
    for (uint16_t i = 0; i < window_width; i++)
        output << "#";
    // Stat header
    output << std::endl << "#";
    for (uint16_t i = 0; i < ((window_width - 11) / 2); i++)
        output << " ";
    output << "STATISTICS";
    for (uint16_t i = 0; i < ((window_width - 11) / 2); i++)
        output << " ";
    output << "#" << std::endl;
    // Total sims
    output << "# Visits" << std::setw(window_width - 8) << std::setfill(' ') << "#" << std::endl; 
    if (node->parent)
        output << "# Parent:" << std::setw(window_width - 11) << std::setfill(' ') << int(node->parent->visits) << " #" << std::endl;
    output << "# Current:" << std::setw(window_width - 12) << std::setfill(' ') << int(node->visits) << " #" << std::endl;
    // Evaluations
    output << "# Evaluations" << std::setw(window_width - 13) << std::setfill(' ') << "#" << std::endl; 
    if (node->parent)
        output << "# Policy:" << std::setw(window_width - 11) << std::setfill(' ') << node->prior_propability << " #" << std::endl;
    output << "# Value:" << std::setw(window_width - 10) << std::setfill(' ') << node->value << " #" << std::endl;
    output << "# Mean Value:" << std::setw(window_width - 15) << std::setfill(' ') << node->meanEvaluation() << " #" << std::endl;
    
    for (uint16_t i = 0; i < window_width; i++)
        output << "#";
    output << std::endl;

    // Print move history
    Node* running_node = node;
    output << "Move history: ";

    while (running_node->parent)
    {
        uint8_t x, y;
        Utils::indexToCords(running_node->parent_action, x, y);
        output << "[" << int(x) << "," << int(y) << "];";
        running_node = running_node->parent;
    }

    output << std::endl;

    return output.str();
}