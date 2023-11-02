#include "Node.h"

Node::Node(State* state, Node* parent, uint16_t parent_action)
    : parent(parent), parent_action(parent_action), state(state), visits(0), network_status(0)
{  
    memset(results, 0, sizeof(uint32_t) * 3);
    untried_actions = state->getPossible();
}

Node::Node(State* state)
    : Node(state, nullptr, uint16_t(-1))
{   }

Node::Node()
    : Node(new State())
{   }

Node::~Node()
{
    delete state;
}

void Node::setModelOutput(std::tuple<torch::Tensor, torch::Tensor> input)
{
    // Extract each heads output
    torch::Tensor policy = std::get<0>(input);
    torch::Tensor value = std::get<1>(input);

    // Assign value
    evaluation = value.detach().item<float>();
    // Normaize for black is +1 white -1
    if (state->nextColor())
        evaluation *= -1;

    // Assign policy values
    for (int i = 0; i < BoardSize * BoardSize; i++)
        policy_evaluations[i] = policy[i].detach().item<float>();

    uint8_t result = 2;
    // If node is terminal use actual playout result
    if (isTerminal())
    {
        result = getResult();
    }
    else
    {
        if (evaluation > -ValueConfidenceBound)
            result = 0;
        else if (evaluation < ValueConfidenceBound)
            result = 1;
    }


    
    backpropagate(result);

    network_status = true;
}

void Node::removeFromUntried(uint16_t action)
{
    untried_actions.erase(
            std::remove(untried_actions.begin(), untried_actions.end(), action), untried_actions.end()
        );
}

bool Node::getNetworkStatus()
{
    return network_status;
}

Node* Node::expand()
{
    if (!network_status)
    {
        std::cout << "Tried to auto expand node without policy data" << std::endl << std::flush;
        return nullptr;
    }

    // Find highest policy action
    uint16_t action = -1; // initialized as unreachable value
    for (uint16_t possible : untried_actions)
        if (action == uint16_t(-1) || policy_evaluations[action] < policy_evaluations[possible])
            action = possible;

    return expand(action);
}

Node* Node::expand(uint16_t action)
{
    removeFromUntried(action);

    State* resulting_state = new State(state);
    resulting_state->makeMove(action);
    Node* child = new Node(resulting_state, this, action);

    children.push_back(child);
    return child;
}

void Node::backpropagate(uint8_t result)
{
    visits++;
    results[result] += 1;

    // Stop at root
    if (parent)
        parent->backpropagate(result);
}

float Node::meanEvaluation()
{
    if (visits == 0)
    {
        std::cout << "[Node][W]: Tried to get meanEvaluation from node with 0 visits?" << std::endl << std::flush;
        return 0;
    }

    if (state->nextColor())
        return float(results[0] - results[1]) / float(visits);
    else
        return float(results[1] - results[0]) / float(visits);  
}

Node* Node::bestChild()
{
    Node* best_child = nullptr;
    float best_result = -100.0;
    float log_visits = 2 * std::log(visits);
    float result, value, exploration, policy;

    // Get child with best value
    for (Node* child : children)
    {
        // Calculate value
        value = ValueBias * child->meanEvaluation();
        exploration = ExplorationBias * std::sqrt(log_visits / float(child->visits));
        policy = PolicyBias * child->getPolicyValue();
        result = value + exploration + policy;

        if (result > best_result)
        {
            best_result = result;
            best_child = child;
        }
    }

    return best_child;
}

bool Node::isTerminal()
{
    return state->isTerminal();
}

Node* Node::absBestChild()
{
    Node* best_child = nullptr;
    uint32_t result;
    uint32_t best_result = 0;

    for (Node* child : children)
    {
        result = child->visits;
        if (result > best_result)
        {
            best_result = result;
            best_child = child;
        }
    }

    return best_child;
}

float Node::getPolicyValue()
{
    return parent->policy_evaluations[parent_action];
}

torch::Tensor Node::nodeToGamestate(Node* node)
{
    // Generate Tensor on CPU
    torch::TensorOptions default_tensor_options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);

    // Init main tensor
    torch::Tensor tensor = torch::zeros({HistoryDepth + 1, 15, 15}, default_tensor_options);

    // State at node
    State* current_state = node->state;

    // Next color tensor
    torch::Tensor next_color;

    if (node->state->nextColor())
        next_color = torch::ones({BoardSize, BoardSize}, default_tensor_options);
    else
        next_color = torch::zeros({BoardSize, BoardSize}, default_tensor_options);
    tensor[0] = next_color;

    // Get last actions from source
    std::deque<uint16_t> move_history;
    Node* running_node = node; 
    for (uint16_t i = 0; i < HistoryDepth - 2; i++)
    {
        if (running_node == nullptr)
        {
            // Is max number which will never be reached
            move_history.push_front(uint16_t(-1));
        }
        else
        {
            move_history.push_front(running_node->parent_action);
            running_node = running_node->parent;
        }
    }

    // Oldest state
    State* history_state;

    // The oldest states of each color
    torch::Tensor history_white = torch::zeros({BoardSize, BoardSize}, default_tensor_options);
    torch::Tensor histroy_black = torch::zeros({BoardSize, BoardSize}, default_tensor_options);
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

    // Embed histroy actions
    for (uint16_t history_move : move_history)
    {
        // If white did HM
        if (color_toggle)
        {
            index_white++;
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

    return tensor;
}

uint8_t Node::getResult()
{
    return state->getResult();
}

// -------------- Utility Code (Not relevant for algorithm) --------------
std::string distribution_helper(Node* child, int max_visits, float max_policy, const std::string& type)
{
    std::ostringstream result;
    result << std::setw(3) << std::setfill(' ');
    if      (type == "VISITS")
        if (max_visits == 0)
            result << 0;
        else
            result << int(float(child->visits) / float(max_visits) * 999);
    else if (type == "VALUE")
        result << int(float(child->evaluation) * 99);
    else if (type == "MEAN")
        result << int(float(child->meanEvaluation()) * 99);
    else if (type == "POLICY")
        result << int(float(child->getPolicyValue()) / max_policy * 999);
    else
        result << "ERR";
    return result.str();
}

std::string distribution(Node* current_node, const std::string& type)
{
    std::ostringstream result;

    Node* parent = current_node->parent;

    result << "\n      <";
    for (uint16_t i = 0; i < BoardSize; i++)
        result << "-";
    result << " " << type;
    result << " DISTRIBUTION ";

    for (uint16_t i = 0; i < BoardSize; i++)
        result << "-";
    result << ">\n    ";

    float max_visits = 0;
    float max_policy = 0;
    for (Node* child : parent->children)
        if (child->visits > max_visits)
            max_visits = child->visits;
    for (Node* child : parent->children)
        if (child->getPolicyValue() > max_policy)
            max_policy = child->getPolicyValue();

    for (uint16_t i = 0; i < BoardSize; i++)
        result << " ---";
    result << "\n";
    for (int y = BoardSize - 1; y >= 0; y--)
    {
        result << std::setw(3) << std::setfill(' ') << y << " ";
        for (int x = 0; x < BoardSize; x++)
        {
            result << "|";
            if (parent->state->isCellEmpty(x, y))
            {
                bool matched = false;
                for (Node* child : parent->children)
                {
                    uint16_t index;
                    Utils::cordsToIndex(index, x, y);
                    if (child->parent_action == index)
                    {
                        matched = true;
                        // If this child was the performed action color
                        if (child == current_node)
                            result << "\033[1;32m";

                        result << distribution_helper(child, max_visits, max_policy, type); 

                        if (child == current_node)
                            result << "\033[0m";
                        break;
                    }        
                }
                if (!matched)
                    result << "   ";
            }
            else
            {
                if (parent->state->getCellValue(x, y))
                    result << "\033[1;31m W \033[0m";
                else
                    result << "\033[1;34m B \033[0m";
            }
        }
        result << "|\n    ";
        for (uint16_t i = 0; i < BoardSize; i++)
            result << " ---";
        result << "\n";
    }

    result << "   ";
    for (uint16_t i = 0; i < BoardSize; i++)
        result << " " << std::setw(3) << std::setfill(' ') << i;

    result << "\n    <";
    for (uint16_t i = 0; i < BoardSize * 2 + 26; i++)
        result << "-";
    result << ">\n";

    return result.str();
}

std::string Node::analytics(Node* node, const std::initializer_list<std::string> distributions)
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
        output << "# Policy:" << std::setw(window_width - 11) << std::setfill(' ') << node->getPolicyValue() << " #" << std::endl;
    output << "# Value:" << std::setw(window_width - 10) << std::setfill(' ') << node->evaluation << " #" << std::endl;
    output << "# Mean Value:" << std::setw(window_width - 15) << std::setfill(' ') << node->meanEvaluation() << " #" << std::endl;
    
    for (uint16_t i = 0; i < window_width; i++)
        output << "#";
    output << std::endl;

    // Print move history
    Node* running_node = node;
    output << "{ Move history: ";

    while (running_node->parent)
    {
        uint8_t x, y;
        Utils::indexToCords(running_node->parent_action, x, y);
        output << "[" << int(x) << "," << int(y) << "];";
        running_node = running_node->parent;
    }

    output << " }" << std::endl;

    if (node->state->nextColor())
        output << "Color: Black" << std::endl;
    else
        output << "Color: White" << std::endl;

    if (node->parent)
    {
        for (const std::string& value : distributions) 
        {
            output << distribution(node, value);
        }
    }


    return output.str();
}

std::string Node::sliceNodeHistory(Node* node, uint8_t depth)
{
    torch::Tensor tensor = nodeToGamestate(node);

    int HD = HistoryDepth;
    if (depth > HD - 2) {
        std::cout << "[Gamestate][W]: Gamestate sliced too deep" << std::endl;
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

    output += "\n   ";
    for (uint16_t i = 0; i < BoardSize; i++)
        output += " ---";
    output += "\n";

    for (int y = 14; y >= 0; y--) {
        output += std::to_string(y) + std::string(3 - std::to_string(y).length(), ' ');
        for (int x = 0; x < 15; x++) {
            output += "|";
            if (blackStones[x][y].item<bool>() == 0 && whiteStones[x][y].item<bool>() == 0) {
                output += "   ";
            } else if (blackStones[x][y].item<bool>() == 1) {
                output += "\033[1;34m B \033[0m";
            } else if (whiteStones[x][y].item<bool>() == 1) {
                output += "\033[1;31m W \033[0m";
            }
        }
        output += "|\n   ";
        for (uint16_t i = 0; i < BoardSize; i++)
            output += " ---";
        output += "\n";
    }

    output += "    ";
    for (uint16_t i = 0; i < BoardSize; i++)
        output += " " + std::to_string(i) + std::string(3 - std::to_string(i).length(), ' ');
    output += "\n";

    return output;
}
