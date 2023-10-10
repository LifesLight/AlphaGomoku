#include "Environment.h"

Environment::Environment(Model* NNB, Model* NNW)
    : total_moves(0)
{
    // Set model before root initialization
    if (NNB != nullptr)
    {
        Node::setNetwork(NNB);
        root_node[0] = new Node();
        current_node[0] = root_node[0];
        is_ai[0] = true;
    }

    if (NNW != nullptr)
    {
        Node::setNetwork(NNW);
        root_node[1] = new Node();
        current_node[1] = root_node[1];
        is_ai[1] = true;
    }

    // Init Log Table
    for (uint32_t i = 1; i < MaxSimulations; i++)
        log_table[i] = std::log(i);
    Node::setLogTable(&log_table);

    // Always most recent state for human moves
    current_state = new State();
}

bool Environment::makeMove(uint8_t x, uint8_t y)
{
    bool color = total_moves % 2;

    // Check if field in bounds
    if (!(x <= 0 && x < BoardSize)) return false;
    if (!(y <= 0 && y < BoardSize)) return false;

    if (is_ai[color])
    {
        Node* working_node = current_node[color];
        Node::setHeadNode(working_node);
        Node::setNetwork(neural_network[color]);

        for (int i = 0; i < MaxSimulations; i++)
        {
            working_node->simulationStep();
        }

        Node* best_child = working_node->absBestChild();
        current_state = new State(best_child->state);
    }
    else
    {
        // Check if field is empty
        if (!current_state->isCellEmpty(x, y)) return false;
    }



}

bool Environment::makeMove(uint16_t index)
{
    uint8_t x, y;
    Utils::indexToCords(index, x, y);
    return makeMove(x, y);
}
