#include "Node.h"

Model* Node::neural_network = nullptr;
float* Node::logTable = nullptr;
Node* Node::head_node = nullptr;

Node::Node(State* state, Node* parent, uint16_t parent_action)
    : parent(parent), parent_action(parent_action), state(state), visits(uint16_t(-1))
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

    for (uint16_t possible_action : possible_actions)
        untried_actions.push_back(std::tuple<uint16_t, float>(possible_action, policy_output[possible_action].item<float>()));

    std::sort(untried_actions.begin(), untried_actions.end(),
    [](const auto& a, const auto& b) {
            return std::get<1>(a) > std::get<1>(b);
        }
    );
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
    for (Node* child : children) delete child;
}

void Node::setNetwork(Model* neural_net)
{
    neural_network = neural_net;
}

void Node::setLogTable(float* log_table)
{
    logTable = log_table;
}

void Node::setHeadNode(Node* head)
{
    head_node = head;
}

void Node::constrain(Node* valid)
{
    bool successfull = false;
    for (Node* child : children)
        if (child != valid)
            delete child;
        else
            successfull = true;
    children.clear();
    if (successfull)  
        children.push_back(valid);
}

Node* Node::expand()
{
    std::tuple<uint16_t, float> next_pair = untried_actions.back();
    untried_actions.pop_back();
    uint16_t action = std::get<0>(next_pair);
    float child_value = std::get<1>(next_pair);

    State* resulting_state = new State(state);
    resulting_state->makeMove(action);
    Node* child = new Node(resulting_state, this, action);

    // Store predicted value of child
    child->prior_propability = child_value;
    children.push_back(child);
    return child;
}

void Node::backpropagate(float evaluation)
{
    visits++;
    summed_evaluation += evaluation;

    // Dont propagate above current head node
    if (parent && this != head_node)
        parent->backpropagate(evaluation);
}

float Node::meanEvaluation(bool turn_color)
{
    if (turn_color)
        return summed_evaluation / float(visits);
    else
        return -(summed_evaluation / float(visits));
}

Node* Node::bestChild()
{
    Node* best_child = nullptr;
    float best_result = -100.0;
    // Precompute
    float log_visits = 2 * logTable[visits];
    bool turn = !state->nextColor();
    float Q_value;
    float result;
    for (Node* child : children)
    {
        Q_value = float(child->meanEvaluation(turn));
        result = Q_value + ExplorationBias * std::sqrt(log_visits / float(child->visits)) * child->prior_propability;
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
    float result;
    float best_result = -100.0;
    // Precompute
    bool turn = !state->nextColor();
    for (Node* child : children)
    {
        result = float(child->meanEvaluation(turn));
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
    float result;
    float best_result = -100.0;
    // Precompute
    bool turn = !state->nextColor();
    
    for (Node* child : children_copy)
    {
        result = float(child->meanEvaluation(turn));
        if (result > best_result)
        {
            best_result = result;
            best_child = child;
        }
    }

    return best_child;
}

void Node::simulationStep()
{
    Node* current = this;
    while (!current->state->isTerminal())
        if (current->untried_actions.size() > 0)
        {
            current = current->expand();
            current->backpropagate(current->value);
            return;
        }
            
        else
            current = current->bestChild();
    current->backpropagate(current->value);
    return;
}

