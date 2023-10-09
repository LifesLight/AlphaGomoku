#include "Gamestate.h"

Gamestate::Gamestate(Node* node)
{
    // Init main tensor
    tensor = torch::zeros({HistoryDepth + 1, 15, 15}, torch::kFloat32);

    // State at node
    State* current_state = node->state;

    // Next color tensor
    torch::Tensor next_color;
    if (node->state->nextColor())
        next_color = torch::ones({BoardSize, BoardSize}, torch::kFloat32);
    else
        next_color = torch::zeros({BoardSize, BoardSize}, torch::kFloat32);
    tensor[0] = next_color;

    // Get last actions from source
    std::deque<uint16_t> move_history;
    Node* running_node = node; 
    for (uint16_t i = 0; i < HistoryDepth - 2; i++)
    {
        if (running_node == nullptr)
        {
            //std::cout << "Found: -" << std::endl;
            // Is max number which will never be reached
            move_history.push_front(uint16_t(-1));
        }
        else
        {
            //std::cout << "Found: " << running_node->parent_action << std::endl;
            move_history.push_front(running_node->parent_action);
            running_node = running_node->parent;
        }
    }

    // Oldest state
    State* history_state;

    // The oldest states of each color
    torch::Tensor history_white = torch::zeros({BoardSize, BoardSize}, torch::kFloat32);
    torch::Tensor histroy_black = torch::zeros({BoardSize, BoardSize}, torch::kFloat32);
    if (running_node != nullptr)
    {
        history_state = running_node->state;
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

    // Indecies into tensor for color
    uint8_t index_black = 1;
    uint8_t index_white = HistoryDepth / 2 + 1;

    tensor[index_black] = histroy_black.clone();
    tensor[index_white] = history_white.clone();

    // Init toggle for what color did what action
    bool color_toggle = current_state->nextColor();

    /*
    std::cout << "Size: " << move_history.size() << std::endl;
    for (uint16_t history_move : move_history)
    {
        std::cout << int(history_move) << std::endl;
    }
    */

    // Embed histroy actions
    for (uint16_t history_move : move_history)
    {
        // If white did HM
        if (color_toggle)
        {
            index_white++;
            //std::cout << "White: " << int(index_white) << std::endl;
            if (history_move != uint16_t(-1))
            {
                uint8_t x, y;
                Utils::indexToCords(history_move, x, y);
                history_white[x][y] = true;
                tensor[index_white] = history_white.clone();
            }
        }
        // If black did HM
        else
        {
            index_black++;
            //std::cout << "Black: " << int(index_black) << std::endl;
            if (history_move != uint16_t(-1))
            {
                uint8_t x, y;
                Utils::indexToCords(history_move, x, y);
                histroy_black[x][y] = true;
                tensor[index_black] = histroy_black.clone();
            }
        }

        color_toggle = !color_toggle;
    }
}

torch::Tensor Gamestate::getTensor()
{
    return tensor;
}

std::string Gamestate::sliceToString(uint8_t depth)
{
    int HD = HistoryDepth;
    if (depth > HD - 2) {
        std::cout << "[Utilities] WARNING: Gamestate sliced too deep" << std::endl;
    }

    std::string output = "";
    int halfDepth = HD / 2;
    torch::Tensor blackStones, whiteStones;

    if (HD == 2) {
        blackStones = tensor[1];
        whiteStones = tensor[2];
    } else {
        int tempDepth = depth / 2;
        int whiteIndex = halfDepth * 2 - tempDepth;
        int blackIndex = halfDepth - tempDepth;

        if (depth % 2 == 1) {
            if (tensor[0][0][0].item<bool>() == 1) {
                blackIndex -= 1;
            } else {
                whiteIndex -= 1;
            }
        }

        blackStones = tensor[blackIndex];
        whiteStones = tensor[whiteIndex];
    }

    output += "   --------------------------------------------------------------\n";
    for (int y = 14; y >= 0; y--) {
        output += std::to_string(y) + " |";
        for (int x = 0; x < 15; x++) {
            if (blackStones[x][y].item<bool>() == 0 && whiteStones[x][y].item<bool>() == 0) {
                output += "   ";
            } else if (blackStones[x][y].item<bool>() == 1) {
                output += " B ";
            } else if (whiteStones[x][y].item<bool>() == 1) {
                output += " W ";
            }
            output += "|";
        }
        output += "\n   --------------------------------------------------------------\n";
    }
    output += "     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14\n";

    return output;
}