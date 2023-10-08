#include "Gamestate.h"

Gamestate::Gamestate(Node* node)
{
    tensor = torch::zeros({HistoryDepth + 1, 15, 15}, torch::kFloat32);
    State* global_state;

    std::vector<uint16_t> move_history;
    move_history.reserve(HistoryDepth);

    Node* current_node = node; 
    for (uint16_t i = 0; i < HistoryDepth; i++)
    {
        if (current_node == nullptr)
            break;
        move_history.push_back(current_node->parent_action);
        current_node = current_node->parent;
    }

    // Take as reference for "global board positions"
    global_state = current_node->state;

    
}