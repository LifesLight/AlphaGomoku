#include "Node.h"

float* Node::logTable = nullptr;

Node::Node(State* state, Node* parent, uint16_t parent_action)
    : parent(parent), parent_action(parent_action), state(state), visits(0)
{   
    is_initialized = false;
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

void Node::runNetwork(Model* neural_network)
{
    std::vector<uint16_t> possible_actions = state->getPossible();

    // Implement model here
    Gamestate gamestate = Gamestate(this);
    torch::Tensor model_input = torch::empty({1, HistoryDepth + 1, BoardSize, BoardSize}, torch::kFloat32);
    model_input[0] = gamestate.getTensor();

    std::tuple<torch::Tensor, torch::Tensor> model_output;
    model_output = neural_network->forward(model_input);

    torch::Tensor policy_output = std::get<0>(model_output)[0];
    torch::Tensor value_output = std::get<1>(model_output)[0];
    value = value_output.item<float>();

    // Normalize for player color, black is positive
    if (state->nextColor())
        value *= -1;

    for (uint16_t possible_action : possible_actions)
        untried_actions.push_back(std::tuple<uint16_t, float>(possible_action, policy_output[possible_action].item<float>()));

    std::sort(untried_actions.begin(), untried_actions.end(),
    [](const auto& a, const auto& b) {
            return std::get<1>(a) < std::get<1>(b);
        }
    );

    is_initialized = true;
}

void Node::setLogTable(float* log_table)
{
    logTable = log_table;
}

Node* Node::expand(Model* neural_network)
{
    if (!is_initialized)
    {
        std::cout << "[Node][E]: Tried to expand uninitialized Node" << std::endl;
        return nullptr;
    }
        
    std::tuple<uint16_t, float> next_pair = untried_actions.back();
    untried_actions.pop_back();
    uint16_t action = std::get<0>(next_pair);
    float child_value = std::get<1>(next_pair);

    State* resulting_state = new State(state);
    resulting_state->makeMove(action);
    Node* child = new Node(resulting_state, this, action);
    child->runNetwork(neural_network);

    // Store predicted value of child
    child->prior_propability = child_value;
    children.push_back(child);
    return child;
}

void Node::backpropagate(float evaluation, Node* head_node)
{
    visits++;
    summed_evaluation += evaluation;

    // Dont propagate above current head node
    if (parent && this != head_node)
        parent->backpropagate(evaluation, head_node);
}

float Node::meanEvaluation()
{
    if (state->nextColor())
        return summed_evaluation / float(visits);
    else
        return -(summed_evaluation / float(visits));  
}

Node* Node::bestChild()
{
    Node* best_child = nullptr;
    float best_result = -100.0;
    float log_visits = 2 * logTable[visits];
    float result, value, exploration, policy;

    for (Node* child : children)
    {
        value = float(child->meanEvaluation());
        exploration = ExplorationBias * std::sqrt(log_visits / float(child->visits));
        policy = PolicyBias * child->prior_propability;

        result = value + exploration + policy;
        //std::cout << "V:" << value << std::endl << "E:" << exploration << std::endl << "P:" << policy << std::endl << std::endl;

        if (result > best_result)
        {
            best_result = result;
            best_child = child;
        }
    }
    return best_child;
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

Node* Node::absBestChild(float confidence_bound)
{
    std::list<Node*> children_copy;
    for (Node* child : children)
       if (float(child->visits) / float(visits) > confidence_bound)
            children_copy.push_back(child);
 
    Node* best_child = nullptr;
    uint32_t result;
    uint32_t best_result = 0;
    
    for (Node* child : children_copy)
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

void Node::simulationStep(Model* neural_network)
{
    Node* current = this;
    while (!current->state->isTerminal())
    {
        if (current->untried_actions.size() > 0)
        {
            current = current->expand(neural_network);
            // this is head node
            current->backpropagate(current->value, this);
            return;
        }
        else
        {
            current = current->bestChild();
        }
    }

    // this is head node
    current->backpropagate(current->value, this);
    return;
}




// Analytic stuff
std::string distribution_helper(Node* child, int max_visits, float max_policy, const std::string& type)
{
    std::ostringstream result;
    result << std::setw(3) << std::setfill(' ');
    if      (type == "VISITS")
        result << int(float(child->visits) / max_visits * 999);
    else if (type == "VALUE")
        result << int(float(child->value) * 100);
    else if (type == "MEAN")
        result << int(float(-child->meanEvaluation()) * 100);
    else if (type == "POLICY")
        result << int(float(child->prior_propability) / max_policy * 999);
    else
        result << "ERR";
    return result.str();
}

std::string distribution(Node* parent, const std::string& type)
{
    std::ostringstream result;

    result << "\n      <";
    for (uint16_t i = 0; i < BoardSize; i++)
        result << "-";
    result << " " << type;
    result << " DISTRIBUTION ";

    for (uint16_t i = 0; i < BoardSize; i++)
        result << "-";
    result << ">\n   ";

    float max_visits = 0;
    float max_policy = 0;
    for (Node* child : parent->children)
        if (child->visits > max_visits)
            max_visits = child->visits;
    for (Node* child : parent->children)
        if (child->prior_propability > max_policy)
            max_policy = child->prior_propability;

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
                        result << distribution_helper(child, max_visits, max_policy, type); 
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
    if (node->children.size() != 1)
        std::cout << "[Node][W]: Analytics is missing chilren, did you free memory before calling?" << std::endl;
        
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

    for (const std::string& value : distributions) 
    {
        std::cout << distribution(node->parent, value);
    }

    return output.str();
}
