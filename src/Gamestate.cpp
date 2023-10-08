#include "Gamestate.h"

Gamestate::Gamestate(Node* node)
{
    // Init main tensor
    tensor = torch::empty({HistoryDepth + 1, 15, 15}, torch::kFloat32);

    // Next color tensor
    torch::Tensor next_color;
    if (node->state->nextColor())
        next_color = torch::ones({BoardSize, BoardSize}, torch::kFloat32);
    else
        next_color = torch::zeros({BoardSize, BoardSize}, torch::kFloat32);
    tensor[0] = next_color;

    // Get last actions from source
    std::deque<uint16_t> move_history(HistoryDepth);
    Node* current_node = node; 
    for (uint16_t i = 0; i < HistoryDepth; i++)
    {
        if (current_node == nullptr)
            break;
        move_history.push_front(current_node->parent_action);
        current_node = current_node->parent;
    }

    // Oldest state
    State* history_state;

    // The oldest states of each color
    torch::Tensor history_white = torch::zeros({BoardSize, BoardSize}, torch::kFloat32);
    torch::Tensor histroy_black = torch::zeros({BoardSize, BoardSize}, torch::kFloat32);;
    if (current_node != nullptr)
    {
        history_state = current_node->state;
        for (uint8_t x = 0; x < BoardSize; x++)
            for (uint8_t y = 0; y < BoardSize; y++)
            {
                uint8_t cell_value = history_state->getCellValue(x, y);
                if (cell_value == 0)
                    histroy_black[x][y] = true;
                else if (cell_value == 1)
                    history_white[x][y] = true;
            }
    }

    uint8_t index_black = 1;
    uint8_t index_white = HistoryDepth / 2 + 1;

    // Write oldest states for each color
    tensor[index_black] = histroy_black;
    tensor[index_white] = history_white;

    // Init toggle for what color did what action
    bool color_toggle = 0;
    if (history_state != nullptr)
        color_toggle = history_state->nextColor();

    // Increment when history is too shallow
    for (uint8_t i = 0; i < (HistoryDepth - move_history.size() - 2) / 2; i++)
    {
        index_black++;
        index_white++;
    }

    for (uint8_t i = 0; i < HistoryDepth - 2; i++)
    {
        // Todo cool index toggle

        torch::Tensor current_board = torch::empty({BoardSize, BoardSize}, torch::kFloat32);
        if (color_toggle)
        {
            torch::copy(current_board, histroy_black);
            index_black++;

        }
        else
        {
            torch::copy(current_board, history_white);
            index_white++;

        }



        color_toggle = !color_toggle;
    }
}