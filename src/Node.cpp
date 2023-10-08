#include "Node.h"


Node::Node(State* state, Node* parent, uint16_t parent_action)
    : parent(parent), parent_action(parent_action), state(state), visits(0)
{
    memset(results, 0, sizeof(uint32_t) * 3);
    std::vector<uint16_t> possible_actions = state->getPossible();

    // Implement model here
    for (uint16_t possible_action : possible_actions)
        untried_actions.push_back(std::tuple<uint16_t, float>(possible_action, 0));
        
    std::shuffle(std::begin(untried_actions), std::end(untried_actions), rng);
}

Node::Node(State* state)
    : Node(new State(state), nullptr, 0)
{   }

Node::Node()
    : Node(new State())
{   }

Node::~Node()
{
    delete state;
    for (Node* child : children) delete child;
}

Node* Node::expand()
{
    std::tuple<uint16_t, float> action = untried_actions.back();
    uint16_t index = std::get<0>(action);
    float child_value = std::get<1>(action);

    untried_actions.pop_back();
    State* resulting_state = new State(state);
    resulting_state->makeMove(index);
    Node* child = new Node(resulting_state, this, index);

    // Store predicted value of child
    child->value = child_value;
    children.push_back(child);
    return child;
}

void Node::rollout()
{
    State simulation_state = State(state);
    std::uniform_int_distribution<std::mt19937::result_type> distribution(0, untried_actions.size());
    uint16_t index = distribution(rng);
    while (!simulation_state.isTerminal())
    {
        simulation_state.makeMove(std::get<0>(untried_actions[index % untried_actions.size()]));
        index++;
    }
    backpropagate(simulation_state.result);
}

void Node::backpropagate(uint8_t value)
{
    visits++;
    results[value]++;
    if (parent)
        parent->backpropagate(value);
}

int32_t Node::qDelta(bool turn)
{
    if (turn)   return results[0] - results[1];
    else        return results[1] - results[0];
}

Node* Node::bestChild()
{
    Node* best_child = nullptr;
    FloatPrecision best_result = -100.0;
    // Precompute
    FloatPrecision log_visits = 2 * logTable[visits];
    bool turn = state->empty % 2;
    FloatPrecision Q_value;
    FloatPrecision result;
    for (Node* child : children)
    {
        Q_value = FloatPrecision(child->qDelta(turn)) / FloatPrecision(child->visits);
        result = Q_value + ExplorationBias * std::sqrt(log_visits / FloatPrecision(child->visits));
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
    bool turn = state->empty % 2;
    for (Node* child : children)
    {
        FloatPrecision Q_value = FloatPrecision(child->qDelta(turn)) / FloatPrecision(child->visits);
        result = Q_value;
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
    bool turn = state->empty % 2;
    
    for (Node* child : children_copy)
    {
        FloatPrecision Q_value = FloatPrecision(child->qDelta(turn)) / FloatPrecision(child->visits);
        result = Q_value;
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

