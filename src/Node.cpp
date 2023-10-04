#include "Node.h"


Node::Node(State state, Node* parent, uint16_t parent_action)
    : parent(parent), parent_action(parent_action), state(state), visits(0)
{
    memset(results, 0, sizeof(uint32_t) * 3);
    untried_actions = state.getPossible();
    std::shuffle(std::begin(untried_actions), std::end(untried_actions), rng);
}

Node::Node(State state)
    : Node(state, nullptr, 0)
{   }

Node::Node()
    : Node(new State())
{   }

Node::~Node()
{
    for (Node* child : children) delete child;
}

Node* Node::expand()
{
    uint16_t index = untried_actions.back();
    untried_actions.pop_back();
    State resulting_state(state);
    resulting_state.makeMove(index);
    Node* child = new Node(resulting_state, this, index);
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
        simulation_state.makeMove(untried_actions[index % untried_actions.size()]);
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
    bool turn = state.empty % 2;
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

Node* Node::policy()
{
    Node* current = this;
    while (!current->state.isTerminal())
        if (current->untried_actions.size() > 0)
            return current->expand();
        else
            current = current->bestChild();
    return current;
}

