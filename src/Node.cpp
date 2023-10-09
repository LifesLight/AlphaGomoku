#include "Node.h"


Node::Node(State* state, Node* parent, uint16_t parent_action, Model* neural_network)
    : parent(parent), parent_action(parent_action), state(state), visits(0), neural_network(neural_network)
{
    std::vector<uint16_t> possible_actions = state->getPossible();

    // Implement model here
    Gamestate gamestate = Gamestate(this);
    torch::Tensor model_input = torch::empty({1, HistoryDepth + 1, BoardSize, BoardSize}, torch::kFloat32);
    model_input[0] = gamestate.getTensor();

    std::tuple<torch::Tensor, float> model_output;
    model_output = neural_network->forward(model_input);

    value = std::get<1>(model_output);
    torch::Tensor policy_output = std::get<0>(model_output)[0];

    for (uint16_t possible_action : possible_actions)
        untried_actions.push_back(std::tuple<uint16_t, float>(possible_action, policy_output[possible_action].item<float>()));

    std::sort(untried_actions.begin(), untried_actions.end(),
        [](const auto& a, const auto& b) {
            return std::get<1>(a) > std::get<1>(b);
        }
    );

    backpropagate(value);
}

Node::Node(State* state, Model* neural_network)
    : Node(new State(state), nullptr, uint16_t(-1), neural_network)
{   }

Node::Node(Model* neural_network)
    : Node(new State(), neural_network)
{   }

Node::~Node()
{
    delete state;
    for (Node* child : children) delete child;
}

Node* Node::expand()
{
    std::tuple<uint16_t, float> next_pair = untried_actions.back();
    untried_actions.pop_back();
    uint16_t action = std::get<0>(next_pair);
    float child_value = std::get<1>(next_pair);

    State* resulting_state = new State(state);
    resulting_state->makeMove(action);
    Node* child = new Node(resulting_state, this, action, neural_network);

    // Store predicted value of child
    child->prior_propability = child_value;
    children.push_back(child);
    return child;
}

void Node::backpropagate(float evaluation)
{
    visits++;
    summed_evaluation += evaluation;
    if (parent)
        parent->backpropagate(evaluation);
}

float Node::meanEvaluation(bool turn_color)
{
    if (!turn_color)
        return summed_evaluation / float(visits);
    else
        return -(summed_evaluation / float(visits));
}

Node* Node::bestChild()
{
    Node* best_child = nullptr;
    FloatPrecision best_result = -100.0;
    // Precompute
    FloatPrecision log_visits = 2 * logTable[visits];
    bool turn = !state->nextColor();
    FloatPrecision Q_value;
    FloatPrecision result;
    for (Node* child : children)
    {
        Q_value = FloatPrecision(child->meanEvaluation(turn));
        result = Q_value + ExplorationBias * std::sqrt(log_visits / FloatPrecision(child->visits)) * child->prior_propability;
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
    FloatPrecision result;
    FloatPrecision best_result = -100.0;
    // Precompute
    bool turn = !state->nextColor();
    for (Node* child : children)
    {
        result = FloatPrecision(child->meanEvaluation(turn));
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
       if (FloatPrecision(child->visits) / FloatPrecision(visits) > confidence_bound)
            children_copy.push_back(child);
 
    Node* best_child = nullptr;
    FloatPrecision result;
    FloatPrecision best_result = -100.0;
    // Precompute
    bool turn = !state->nextColor();
    
    for (Node* child : children_copy)
    {
        result = FloatPrecision(child->meanEvaluation(turn));
        if (result > best_result)
        {
            best_result = result;
            best_child = child;
        }
    }

    return best_child;
}

Node* Node::policy()
{
    Node* current = this;
    while (!current->state->isTerminal())
        if (current->untried_actions.size() > 0)
            return current->expand();
        else
            current = current->bestChild();
    return current;
}

