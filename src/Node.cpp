 #include "Node.h"

Node::Node(State* state, Node* parent)
    : parent(parent), state(state), network_status(0)
{
    temp_data = new NodeData();
    temp_data->untried_actions = state->getPossible();
    temp_data->visits = 0;
    temp_data->summed_evaluation = 0.0f;
}

Node::Node(State* state)
    : Node(state, nullptr)
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

void Node::reset()
{
    temp_data->untried_actions = state->getPossible();
    for (Node* child : children)
        delete child;
    children.clear();
}

void Node::deleteState()
{
    delete state;
    state = nullptr;
}

float Node::getValueHeadEval()
{
    if (temp_data)
        return temp_data->evaluation;
    else
    {
        Log::log(LogLevel::ERROR, "Tried to get value head eval from shrunk node", "NODE");
        return 0;
    }
}

float Node::getSummedEvaluation()
{
    if (temp_data)
        return temp_data->summed_evaluation;
    else
    {
        Log::log(LogLevel::ERROR, "Tried to get summed value from shrunk node", "NODE");
        return 0;
    }
}

uint32_t Node::getVisits()
{
    if (temp_data)
        return temp_data->visits;
    else
    {
        Log::log(LogLevel::ERROR, "Tried to get visits from shrunk node", "NODE");
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
            Log::log(LogLevel::ERROR, "Tried to get policy value form shrunk node", "NODE");
            return 0;
        }
    }
    else
    {
        Log::log(LogLevel::ERROR, "Tried to get policy value form node without net data", "NODE");
        return 0;
    }
}

bool Node::isFullyExpanded()
{
    if (isShrunk())
        return true;
    // Adds possibility to clamp exploration to n best initial guesses
    #if BranchingLimit > 0
    if (children.size() > BranchingLimit || getUntriedActions().size() == 0)
        return true;
    #else
    if (getUntriedActions().size() == 0)
        return true;
    #endif
    return false;
}

std::vector<index_t>& Node::getUntriedActions()
{
    if (temp_data)
        return temp_data->untried_actions;
    else
    {
        Log::log(LogLevel::ERROR, "Tried to get untried actions from shrunk node", "NODE");
        // Will crash but whatever
        return temp_data->untried_actions;
    }
}

index_t Node::getParentAction()
{
    if (parent)
        return state->last;
    else
        return index_t(-1);
}

void Node::shrinkNode()
{
    delete temp_data;
    temp_data = nullptr;
    children.shrink_to_fit();
}

bool Node::isShrunk()
{
    return temp_data == nullptr;
}

void Node::removeNodeFromChildren(Node* node)
{
    Utils::eraseFromVector(children, node);
    // TODO Maybe obsolete ?
    children.shrink_to_fit();
}

bool Node::getNextColor()
{
    return state->getNextColor();
}

float Node::getNodesPolicyEval()
{
    if (parent)
        return parent->getPolicyValue(getParentAction());
    else
    {
        Log::log(LogLevel::ERROR, "Tried to get parent-less nodes policy eval", "NODE");
        return 0.0f;
    }
}

void Node::setModelOutput(torch::Tensor policy, torch::Tensor value)
{
    if (temp_data == nullptr)
    {
        Log::log(LogLevel::ERROR, "Tried to assign net data to shrunk node", "NODE");
        return;
    }
    // Disable gradients for this scope
    torch::NoGradGuard no_grad_guard;

    // Assign value
    float evaluation = value.item<float>();
    // Normaize for black is -1 white +1
    #ifdef DEBUG_INVERT_MODEL_COLORS
    if (getNextColor())
        evaluation *= -1;
    #else
    if (!getNextColor())
        evaluation *= -1;
    #endif
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
    std::vector<index_t>& untried = getUntriedActions();
    Utils::eraseFromVector(untried, action);
}

Node* Node::expand()
{
    if (!network_status)
    {
        Log::log(LogLevel::ERROR, "Tried to auto expand node without policy data", "NODE");
        return nullptr;
    }

    if (!temp_data)
    {
        Log::log(LogLevel::ERROR, "Tried to auto expand shrunk node", "NODE");
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
    Node* child = new Node(resulting_state, this);

    children.push_back(child);

    return child;
}

void Node::callBackpropagate()
{
    if (!getNetworkStatus())
    {
        Log::log(LogLevel::ERROR, "Tried to call backpropagate on node without network data", "NODE");
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
        value = Config::valueBias() * child->getMeanEvaluation();
        exploration = Config::explorationBias() * std::sqrt(log_visits / float(child->getVisits()));
        policy = Config::policyBias() * child->getNodesPolicyEval();
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
            Log::log(LogLevel::WARNING, "Got abs best child from shrunk node, which had more then one child", "NODE");
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

StateResult Node::getResult()
{
    return state->getResult();
}

float Node::getProcessedEval()
{
    if (isShrunk())
    {
        Log::log(LogLevel::ERROR, "Tried to get processed eval from shrunk node", "NODE");
        return 0;
    }

    return valueProcessor(getValueHeadEval());
}

// ------------ Value aggregation ------------

float Node::getMeanEvaluation()
{
    if (getVisits() == 0)
    {
        Log::log(LogLevel::WARNING, "Tried to get meanEvaluation from node with 0 visits", "NODE");
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
    StateResult result = getResult();
    switch (result)
    {
        case StateResult::BLACKWIN:
            return -1.0f;
        case StateResult::WHITEWIN:
            return 1.0f;
        case StateResult::DRAW:
            return 0.0f;
        case StateResult::NONE:
            break;
    }
    return normalized_value;
}

// -------------- Utlility Code --------------

std::vector<index_t> Node::getMoveHistory()
{
    std::vector<index_t> history;
    Node* node = this;
    while (node->parent)
    {
        history.push_back(node->getParentAction());
        node = node->parent;
    }
    std::reverse(history.begin(), history.end());
    return history;
}

torch::Tensor Node::nodeToGamestate(Node* node)
{
    return nodeToGamestate(node, Config::torchScalar());
}

torch::Tensor Node::nodeToGamestate(Node* node, torch::ScalarType dtype)
{
    // Disable gradients for this scope
    torch::NoGradGuard no_grad_guard;

    // Generate Tensor on CPU
    torch::TensorOptions default_tensor_options = torch::TensorOptions().device(Config::torchHostDevice()).dtype(dtype).requires_grad(false);

    // Init main tensor
    torch::Tensor tensor = torch::zeros({Config::historyDepth() + 1, 15, 15}, default_tensor_options);

    // State at node
    State* current_state = node->state;

    // Next color tensor
    torch::Tensor next_color;

    #ifdef DEBUG_INVERT_MODEL_COLORS
    if (!node->getNextColor())
        next_color = torch::ones({BoardSize, BoardSize}, default_tensor_options);
    else
        next_color = torch::zeros({BoardSize, BoardSize}, default_tensor_options);
    #else
    if (node->getNextColor())
        next_color = torch::ones({BoardSize, BoardSize}, default_tensor_options);
    else
        next_color = torch::zeros({BoardSize, BoardSize}, default_tensor_options);
    #endif
    tensor[0] = next_color;

    // Get last actions from source
    std::vector<index_t> move_history;
    move_history.reserve(Config::historyDepth() - 2);

    Node* running_node = node; 
    for (index_t i = 0; i < Config::historyDepth() - 2; i++)
    {
        if (running_node == nullptr)
        {
            // Is max number which will never be reached
            move_history.push_back(index_t(-1));
        }
        else
        {
            move_history.push_back(running_node->getParentAction());
            running_node = running_node->parent;
        }
    }
    std::reverse(move_history.begin(), move_history.end());

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
    uint8_t index_white = Config::historyDepth() / 2 + 1;

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

// -------------- Analysis Code --------------
 
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

    result << "\n      ᐊ";
    for (int i = 0; i < BoardSize; i++)
        result << "═";
    result << "╡ " << type;
    result << " DISTRIBUTION ╞";

    for (int i = 0; i < BoardSize; i++)
        result << "═";
    result << "ᐅ\n";

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

    std::vector<std::vector<std::string>> values;

    for (int x = 0; x < BoardSize; x++)
    {
        std::vector<std::string> line;
        for (int y = 0; y < BoardSize; y++)
        {
            std::stringstream cell;
            if (parent->state->isCellEmpty(x, y))
            { 
                bool matched = false;
                for (Node* child : parent->children)
                {
                    index_t index;
                    Utils::cordsToIndex(index, x, y);
                    if (child->getParentAction() == index)
                    {
                        matched = true;
                        // If this child was the performed action color
                        if (child == current_node)
                            cell << "\033[1;32m";
                        cell << distribution_helper(child, max_value, type, child != current_node); 
                        cell << "\033[0m";
                        break;
                    }
                }
                if (!matched)
                    cell << "   ";
            }
            else
            {
                if (!parent->state->getCellValue(x, y))
                    cell << Style::bsc() << Style::bsu() << "\033[0m";
                else
                    cell << Style::wsc() << Style::wsu() << "\033[0m";
            }
            line.push_back(cell.str());
        }
        values.push_back(line);
    }

    result << Utils::renderGamegrid(values);

    result << "\n    ᐊ";
    for (int i = 0; i < BoardSize * 2 + 26; i++)
        result << "═";
    result << "ᐅ\n";

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
        Utils::indexToCords(running_node->getParentAction(), x, y);
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
