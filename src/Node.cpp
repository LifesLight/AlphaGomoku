#include "Node.h"

Node::Node(State* state, Node* parent, index_t parent_action)
    : parent(parent), parent_action(parent_action), state(state), network_status(0)
{
    temp_data = new NodeData();
    temp_data->untried_actions = state->getPossible();
    temp_data->visits = 0;
    temp_data->summed_evaluation = 0.0f;
}

Node::Node(State* state)
    : Node(state, nullptr, index_t(-1))
{   }

Node::Node()
    : Node(new State())
{   }

Node::~Node()
{
    delete state;
    delete temp_data;

    // Recursively delete all children
    for (Node* child : children)
        delete child;
}

#pragma region DataInterface

float Node::getValueHeadEval()
{
    if (temp_data)
        return temp_data->evaluation;
    else
    {
        std::cout << "[Node][E]: Tried to get value head eval from shrunk node" << std::endl << std::flush;
        return 0;
    }
}

float Node::getSummedEvaluation()
{
    if (temp_data)
        return temp_data->summed_evaluation;
    else
    {
        std::cout << "[Node][E]: Tried to get summed value from shrunk node" << std::endl << std::flush;
        return 0;
    }
}

uint32_t Node::getVisits()
{
    if (temp_data)
        return temp_data->visits;
    else
    {
        std::cout << "[Node][W]: Tried to get visits from shrunk node" << std::endl << std::flush;
        return 0;
    }
}

float Node::getPolicyValue(index_t move)
{
    if (network_status)
    {
        if (temp_data)
        {
            torch::NoGradGuard no_grad_guard;
            return temp_data->policy_evaluations[move].item<float>();
        }
        else
        {
            std::cout << "[Node][E]: Tried to get policy value form shrunk node" << std::endl << std::flush;
            return 0;
        }
    }
    else
    {
        std::cout << "[Node][E]: Tried to get policy value form node without net data" << std::endl << std::flush;
        return 0;
    }
}

bool Node::isFullyExpanded()
{
    if (isShrunk())
        return true;
    if (getUntriedActions().size() == 0)
        return true;
    return false;
}

std::deque<index_t>& Node::getUntriedActions()
{
    if (temp_data)
        return temp_data->untried_actions;
    else
    {
        std::cout << "[Node][E]: Tried to get untried actions from shrunk node" << std::endl << std::flush;
        // Will crash but whatever
        return temp_data->untried_actions;
    }
}

#pragma endregion

void Node::shrinkNode()
{
    delete temp_data;
    temp_data = nullptr;
}

bool Node::isShrunk()
{
    return temp_data == nullptr;
}

void Node::removeNodeFromChildren(Node* node)
{
    children.remove(node);
}

bool Node::getNextColor()
{
    return state->getNextColor();
}

float Node::getNodesPolicyEval()
{
    if (parent)
        return parent->getPolicyValue(parent_action);
    else
    {
        std::cout << "[Node][E]: Tried to get parent less nodes policy eval" << std::endl << std::flush;
        return 0.0f;
    }
}

void Node::setModelOutput(torch::Tensor policy, torch::Tensor value)
{
    if (temp_data == nullptr)
    {
        std::cout << "[Node][E]: Tried to assign net data to shrunk node" << std::endl << std::flush;
        return;
    }
    // Disable gradients for this scope
    torch::NoGradGuard no_grad_guard;

    // Assign value
    float evaluation = value.item<float>();
    // Normaize for black is -1 white +1
    if (!getNextColor())
        evaluation *= -1;
    temp_data->evaluation = evaluation;

    // Store policy values
    temp_data->policy_evaluations = policy;

    // Tell node that it has network data
    network_status = true;

    // Inital backprop
    callBackpropagate();
}

void Node::removeFromUntried(index_t action)
{
    std::deque<index_t>& untried = getUntriedActions();
    untried.erase(
            std::remove(untried.begin(), untried.end(), action), untried.end()
        );
}

Node* Node::expand()
{
    if (!network_status)
    {
        std::cout << "Tried to auto expand node without policy data" << std::endl << std::flush;
        return nullptr;
    }

    if (!temp_data)
    {
        std::cout << "Tried to auto expand shrunk node" << std::endl << std::flush;
        return nullptr;
    }


    // Find highest policy action
    index_t action = -1; // initialized as unreachable value
    float action_val = 0;
    for (index_t possible : getUntriedActions())
    {
        float policy_val = getPolicyValue(possible);
        if (action == index_t(-1) || action_val < policy_val)
        {
            action_val = policy_val;
            action = possible;
        }
    }

    return expand(action);
}

Node* Node::expand(index_t action)
{
    removeFromUntried(action);

    State* resulting_state = new State(state);
    resulting_state->makeMove(action);
    Node* child = new Node(resulting_state, this, action);

    children.push_back(child);
    return child;
}

void Node::callBackpropagate()
{
    if (!getNetworkStatus())
    {
        std::cout << "[Node][W]: Tried to call Backpropagate on node without network data" << std::endl << std::flush;
        return;
    }

    float value = valueProcessor(getValueHeadEval());
    backpropagate(value);
}

Node* Node::bestChild()
{
    Node* best_child = nullptr;
    float best_result = -100.0;
    float log_visits = 2 * std::log(getVisits());
    float result, value, exploration, policy;

    // Get child with best value
    for (Node* child : children)
    {
        // Calculate value
        value = ValueBias * child->getMeanEvaluation();
        exploration = ExplorationBias * std::sqrt(log_visits / float(child->getVisits()));
        policy = PolicyBias * child->getNodesPolicyEval();
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

bool Node::getNetworkStatus()
{
    return network_status;
}

Node* Node::absBestChild()
{
    // If node is shrunk it should usually only have one remaining child, so default to that one
    if (isShrunk())
    {
        if (children.size() > 1)
            std::cout << "[Node][W]: Got abs best child from shrunk node, which had more then one child" << std::endl << std::flush;
        return children.front();
    }

    Node* best_child = nullptr;
    uint32_t result;
    uint32_t best_result = 0;

    for (Node* child : children)
    {
        result = child->getVisits();

        if (result > best_result)
        {
            best_result = result;
            best_child = child;
        }
    }

    return best_child;
}

uint8_t Node::getResult()
{
    return state->getResult();
}

// ------------ Value aggregation ------------
#pragma region

float Node::getMeanEvaluation()
{
    if (getVisits() == 0)
    {
        std::cout << "[Node][W]: Tried to get meanEvaluation from node with 0 visits?" << std::endl << std::flush;
        return 0;
    }

    if (getNextColor())
        return -getSummedEvaluation() / getVisits();
    else
        return getSummedEvaluation() / getVisits();  
}

void Node::backpropagate(float eval)
{
    temp_data->visits++;
    temp_data->summed_evaluation += eval;

    // Stop at root
    if (parent)
        if (!parent->isShrunk())
            parent->backpropagate(eval);
}

float Node::valueProcessor(float normalized_value)
{
    if (isTerminal())
    {
        uint8_t result = getResult();
        if (result == 2)
            normalized_value = 0.0f;
        else if (result == 0)
            normalized_value = -1.0f;
        else
            normalized_value = 1.0f;
    }

    return normalized_value;
}

#pragma endregion

// -------------- Utlility Code --------------
#pragma region

std::deque<index_t> Node::getMoveHistory()
{
    std::deque<index_t> history;
    Node* node = this;
    while (node->parent)
    {
        history.push_front(node->parent_action);
        node = node->parent;
    }
    return history;
}

torch::Tensor Node::nodeToGamestate(Node* node)
{
    return nodeToGamestate(node, TorchDefaultScalar);
}

torch::Tensor Node::nodeToGamestate(Node* node, torch::ScalarType dtype)
{
    // Disable gradients for this scope
    torch::NoGradGuard no_grad_guard;

    // Generate Tensor on CPU
    torch::TensorOptions default_tensor_options = torch::TensorOptions().device(TorchDefaultDevice).dtype(dtype);

    // Init main tensor
    torch::Tensor tensor = torch::zeros({HistoryDepth + 1, 15, 15}, default_tensor_options);

    // State at node
    State* current_state = node->state;

    // Next color tensor
    torch::Tensor next_color;

    if (node->getNextColor())
        next_color = torch::ones({BoardSize, BoardSize}, default_tensor_options);
    else
        next_color = torch::zeros({BoardSize, BoardSize}, default_tensor_options);
    tensor[0] = next_color;

    // Get last actions from source
    std::deque<index_t> move_history;
    Node* running_node = node; 
    for (index_t i = 0; i < HistoryDepth - 2; i++)
    {
        if (running_node == nullptr)
        {
            // Is max number which will never be reached
            move_history.push_front(index_t(-1));
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
    bool color_toggle = current_state->getNextColor();

    // Embed histroy actions
    for (index_t history_move : move_history)
    {
        // If white did HM
        if (color_toggle)
        {
            index_white++;
            if (history_move != index_t(-1))
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
            if (history_move != index_t(-1))
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

#pragma endregion

// -------------- Analysis Code --------------
#pragma region 

std::string distribution_helper(Node* child, float max_value, const std::string& type, bool color_me)
{
    std::ostringstream result;
    if      (type == "VISITS")
    {
        result << std::setw(3) << std::setfill(' ');
        if (int(max_value) == 0)
            result << 0;
        else
            result << int(float(child->getVisits()) / max_value * 999);
    }

    // Cases where we need a max value for "max value of type"
    else if (type == "VALUE")
    {
        // Un-normalize
        float value = child->getValueHeadEval() * (1 - child->getNextColor() * 2);
        if (value == max_value && color_me)
            result << "\033[1;33m";
        result << std::setw(3) << std::setfill(' ');
        result << int(value * 99);
    }
    else if (type == "MEAN")
    {
        float value = child->getMeanEvaluation();
        if (value == max_value && color_me)
            result << "\033[1;33m";
        result << std::setw(3) << std::setfill(' ');
        result << int(float(value) * 99);
    }
    else if (type == "POLICY")
    {
        float value = child->getNodesPolicyEval();
        if (value == max_value && color_me)
            result << "\033[1;33m";
        result << std::setw(3) << std::setfill(' ');
        result << int(float(value) / max_value * 999);
    }
    else
        result << "ERR";
    return result.str();
}

std::string distribution(Node* current_node, const std::string& type)
{
    std::ostringstream result;

    Node* parent = current_node->parent;

    result << "\n        <";
    for (int i = 0; i < BoardSize; i++)
        result << "-";
    result << " " << type;
    result << " DISTRIBUTION ";

    for (int i = 0; i < BoardSize; i++)
        result << "-";
    result << ">\n    ";

    float max_value = 0.0f;
    for (Node* child : parent->children)
    {
        float val = 0;
        if (type == "VALUE")
            val = child ->getValueHeadEval() * (1 - child->getNextColor() * 2);
        else if (type == "MEAN")
            val = child->getMeanEvaluation();
        else if (type == "POLICY")
            val = child->getNodesPolicyEval();
        else if (type == "VISITS")
            val = child->getVisits();

        if (val > max_value)
            max_value = val;
    }


    for (int i = 0; i < BoardSize; i++)
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
                    index_t index;
                    Utils::cordsToIndex(index, x, y);
                    if (child->parent_action == index)
                    {
                        matched = true;
                        // If this child was the performed action color
                        if (child == current_node)
                            result << "\033[1;32m";
                        result << distribution_helper(child, max_value, type, child != current_node); 
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
        for (int i = 0; i < BoardSize; i++)
            result << " ---";
        result << "\n";
    }

    result << "   ";
    for (int i = 0; i < BoardSize; i++)
        result << " " << std::setw(3) << std::setfill(' ') << i;

    result << "\n    <";
    for (int i = 0; i < BoardSize * 2 + 26; i++)
        result << "-";
    result << ">\n";

    return result.str();
}

std::string Node::analytics(Node* node, const std::initializer_list<std::string> distributions)
{
    std::ostringstream output;
    output << std::endl;

    // Static window
    int window_width = 40;
    for (int i = 0; i < window_width; i++)
        output << "#";
    // Stat header
    output << std::endl << "#";
    for (int i = 0; i < ((window_width - 11) / 2); i++)
        output << " ";
    output << "STATISTICS";
    for (int i = 0; i < ((window_width - 11) / 2); i++)
        output << " ";
    output << "#" << std::endl;
    // Total sims
    output << "# Visits" << std::setw(window_width - 8) << std::setfill(' ') << "#" << std::endl; 
    if (node->parent)
        output << "# Parent:" << std::setw(window_width - 11) << std::setfill(' ') << int(node->parent->getVisits()) << " #" << std::endl;
    output << "# Current:" << std::setw(window_width - 12) << std::setfill(' ') << int(node->getVisits()) << " #" << std::endl;
    // Evaluations
    output << "# Evaluations" << std::setw(window_width - 13) << std::setfill(' ') << "#" << std::endl; 
    if (node->parent)
        output << "# Policy:" << std::setw(window_width - 11) << std::setfill(' ') << node->getNodesPolicyEval() << " #" << std::endl;
    output << "# Value:" << std::setw(window_width - 10) << std::setfill(' ') << node->getValueHeadEval() << " #" << std::endl;
    output << "# Mean Value:" << std::setw(window_width - 15) << std::setfill(' ') << node->getMeanEvaluation() << " #" << std::endl;
    
    for (int i = 0; i < window_width; i++)
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

    if (node->getNextColor())
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
    return Utils::sliceGamestate(tensor, depth);
}

#pragma endregion