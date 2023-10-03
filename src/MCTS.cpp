#include "State.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <list>
#include <string>
#include <chrono>
#include <random>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <cstring>

#define BIAS 1.4142
#define BATCH_BoardSize 1000
#define MAX_SIMULATIONS 250000000
typedef float PREC;


std::random_device rand_device;
std::mt19937 rng(rand_device());
PREC log_table[MAX_SIMULATIONS];
uint64_t log_table_size = 0;

class NODE;
class STATISTICS;

class STATISTICS
{
public:
    State state;
    uint32_t visits;
    uint32_t results[3];

    STATISTICS()
        : visits(0), state(new State())
    {
        memset(results, 0, sizeof(uint32_t) * 3);
    }

    STATISTICS(State state)
        : visits(0), state(state)
    {
        memset(results, 0, sizeof(uint32_t) * 3);
    }

    STATISTICS(STATISTICS* source)
        : state(State(source->state)), visits(source->visits)
    {
        memcpy(results, source->results, sizeof(uint32_t) * 3);
    }
};

class NODE
{
public:
    NODE* parent;
    uint16_t parent_action;
    STATISTICS* data;
    std::list<NODE*> children;
    std::vector<uint16_t> untried_actions;

    NODE()
        : parent(nullptr), data(new STATISTICS())
    {
        untried_actions = data->state.getPossible();
        std::shuffle(std::begin(untried_actions), std::end(untried_actions), rng);
    }

    NODE(State state)
        : parent(nullptr), data(new STATISTICS(state))
    {
        untried_actions = data->state.getPossible();
        std::shuffle(std::begin(untried_actions), std::end(untried_actions), rng);
    }

    NODE(State state, NODE* parent, uint16_t parent_action)
        : parent(parent), parent_action(parent_action), data(new STATISTICS(state))
    {
        untried_actions = data->state.getPossible();
        std::shuffle(std::begin(untried_actions), std::end(untried_actions), rng);
    }

    NODE(STATISTICS* data, NODE* parent, uint16_t parent_action)
        : parent(parent), parent_action(parent_action), data(data)
    {
        untried_actions = data->state.getPossible();
        std::shuffle(std::begin(untried_actions), std::end(untried_actions), rng);
    }

    NODE(NODE* source)
        : parent(source->parent), parent_action(source->parent_action), data(new STATISTICS(source->data))
    {
        untried_actions.reserve(source->untried_actions.size());
        untried_actions.assign(source->untried_actions.begin(), source->untried_actions.end());
        for (NODE* child : source->children) children.push_back(new NODE(child));
    }

    NODE(NODE* source, NODE* parent)
        : parent(parent), parent_action(source->parent_action), data(new STATISTICS(source->data))
    {
        untried_actions.reserve(source->untried_actions.size());
        untried_actions.assign(source->untried_actions.begin(), source->untried_actions.end());
        for (NODE* child : source->children) children.push_back(new NODE(child));
    }

    ~NODE()
    {
        for (NODE* child : children) delete child;
    }

    NODE* expand()
    {
        uint16_t index = untried_actions.back();
        untried_actions.pop_back();

        State resulting_state(data->state);
        resulting_state.makeMove(index);

        NODE* child;
        STATISTICS* child_stats;

        child_stats = new STATISTICS(resulting_state);
        child = new NODE(child_stats, this, index);
        children.push_back(child);

        return child;
    }

    void rollout()
    {
        State simulation_state = State(data->state);
        std::uniform_int_distribution<std::mt19937::result_type> distribution(0, untried_actions.size());
        uint16_t index = distribution(rng);

        while (!simulation_state.isTerminal())
        {
            simulation_state.makeMove(untried_actions[index % untried_actions.size()]);
            index++;
        }
        backpropagate(simulation_state.result);
    }

    void backpropagate(uint8_t value)
    {
        data->visits++;
        data->results[value]++;
        if (parent)
            parent->backpropagate(value);
    }

    int32_t Q_delta(bool turn)
    {
        if (turn)   return data->results[0] - data->results[1];
        else        return data->results[1] - data->results[0];
    }

    NODE* best_child()
    {
        NODE* best_child = nullptr;
        PREC best_result = -100.0;

        // Precompute
        PREC log_visits = 2 * log_table[data->visits];
        bool turn = data->state.empty % 2;

        PREC Q_value;

        PREC result;
        for (NODE* child : children)
        {

            Q_value = PREC(child->Q_delta(turn)) / PREC(child->data->visits);
            result = Q_value + BIAS * std::sqrt(log_visits / PREC(child->data->visits));
            if (result > best_result)
            {
                best_result = result;
                best_child = child;
            }
        }
        return best_child;
    }

    NODE* policy()
    {
        NODE* current = this;
        while (!current->data->state.isTerminal())
            if (current->untried_actions.size() > 0)
                return current->expand();
            else
                current = current->best_child();
        return current;
    }
};

class HOST
{
public:

    static void init()
    {
        for (uint32_t i = 1; i < BATCH_BoardSize; i++)
            log_table[i] = std::log(i);
        log_table_size = BATCH_BoardSize;
    }

    static State* create()
    {
        return new State();
    }

    static State* create(std::string position)
    {
        State* state = new State();
        for (int i = 0; i < position.length(); i += 2)
        {
            uint16_t x = position[i] - '0';
            uint16_t y = position[i + 1] - '0';
            state->makeMove(y * BoardSize + x);
        }
        return state;
    }

    static void human_move(State* state)
    {
        bool getting_input = true;
        std::string input_x;
        std::string input_y;
        uint16_t x;
        uint16_t y;
        uint16_t index;
        while (getting_input)
        {
            try
            {
                std::cout << "Action X: ";
                std::cin >> input_x;
                std::cout << "Action Y: ";
                std::cin >> input_y;
                x = stoi(input_x);
                y = stoi(input_y);
            }
            catch (const std::exception& e)
            {
                std::cout << "Invalid input format!\n";
                continue;
            }
            index = y * BoardSize + x;
            if ((0 <= x && x < BoardSize) && (0 <= y && y < BoardSize))
            {
                if (!(state->m_array[y] & (BLOCK(1) << x % BoardSize)))
                    getting_input = false;
                else
                    std::cout << "Selected field is occupied!\n";
            }
            else
                std::cout << "Selection outside the field!\n";
        }
        state->makeMove(index);
    }

    static void MCTS_move(State* root_state, std::chrono::milliseconds time, PREC confidence_bound, bool analytics)
    {
        NODE* root = new NODE(root_state);
        // Build Tree
        uint64_t i;
        uint64_t ix = 0;
        auto start = std::chrono::high_resolution_clock::now();
        while (std::chrono::high_resolution_clock::now() - start < time)
        {
            for (i = 0; i < BATCH_BoardSize; i++)
            {
                NODE* node = root->policy();
                node->rollout();
            }
            ix++;
            update(ix);
        }
        MCTS_master(root, root_state, confidence_bound, analytics);
    }

    static void MCTS_move(State* root_state, uint64_t simulations, PREC confidence_bound, bool analytics)
    {
        NODE* root = new NODE(root_state);
        for (uint64_t i = log_table_size; i < simulations; i++)
            log_table[i] = std::log(i);
        if (log_table_size < simulations)
            log_table_size = simulations;
        // Build Tree
        for (uint64_t i = 0; i < simulations; i++)
        {
            NODE* node = root->policy();
            node->rollout();
        }
        MCTS_master(root, root_state, confidence_bound, analytics);
    }

    static void render_sim_distribution(NODE* root)
    {
        std::cout << "\n    <";
        for (uint16_t i = 0; i < BoardSize; i++)
            std::cout << "-";
        std::cout << " SIMULATIONS DISTRIBUTION ";

        for (uint16_t i = 0; i < BoardSize; i++)
            std::cout << "-";
        std::cout << ">\n";

        PREC max_visits = 0;
        for (NODE* child : root->children)
            if (child->data->visits > max_visits)
                max_visits = child->data->visits;

        std::cout << "    ";
        for (uint16_t i = 0; i < BoardSize; i++)
            std::cout << " " << std::to_string(i).append(3 - std::to_string(i).length(), ' ');
        std::cout << "\n   ";
        for (uint16_t i = 0; i < BoardSize; i++)
            std::cout << " ---";
        std::cout << "\n";
        for (uint16_t y = 0; y < BoardSize; y++)
        {
            std::cout << std::to_string(y).append(3 - std::to_string(y).length(), ' ');
            for (uint16_t x = 0; x < BoardSize; x++)
            {
                std::cout << "|";
                if (!(root->data->state.m_array[y] & (BLOCK(1) << x)))
                {
                    for (NODE* child : root->children)
                        if (child->parent_action == (y * BoardSize + x))
                            std::printf("%3d", int(PREC(child->data->visits) / max_visits * 100));
                }

                else if (root->data->state.c_array[y] & (BLOCK(1) << x))
                {
                    if (!(root->data->state.empty % 2))   std::cout << "\033[1;34m o \033[0m";
                    else                std::cout << "\033[1;31m o \033[0m";
                }
                else if (!(root->data->state.c_array[y] & (BLOCK(1) << x)))
                {
                    if (root->data->state.empty % 2)      std::cout << "\033[1;34m o \033[0m";
                    else                std::cout << "\033[1;31m o \033[0m";
                }
            }
            std::cout << "|\n   ";
            for (uint16_t i = 0; i < BoardSize; i++)
                std::cout << " ---";
            std::cout << "\n";
        }

        std::cout << "    <";
        for (uint16_t i = 0; i < BoardSize * 2 + 26; i++)
            std::cout << "-";
        std::cout << ">\n";
    }

    static void render_ucb_distribution(NODE* root)
    {
        std::cout << "\n    <";
        for (uint16_t i = 0; i < BoardSize; i++)
            std::cout << "-";
        std::cout << "     UCB DISTRIBUTION     ";

        for (uint16_t i = 0; i < BoardSize; i++)
            std::cout << "-";
        std::cout << ">\n";

        std::cout << "    ";
        for (uint16_t i = 0; i < BoardSize; i++)
            std::cout << " " << std::to_string(i).append(3 - std::to_string(i).length(), ' ');
        std::cout << "\n   ";
        for (uint16_t i = 0; i < BoardSize; i++)
            std::cout << " ---";
        std::cout << "\n";
        for (uint16_t y = 0; y < BoardSize; y++)
        {
            std::cout << std::to_string(y).append(3 - std::to_string(y).length(), ' ');
            for (uint16_t x = 0; x < BoardSize; x++)
            {
                std::cout << "|";
                if (!(root->data->state.m_array[y] & (BLOCK(1) << x)))
                {
                    for (NODE* child : root->children)
                        if (child->parent_action == (y * BoardSize + x))
                            std::printf("%+3d", int(PREC(child->Q_delta(root->data->state.empty % 2)) / PREC(child->data->visits) * 100));
                }

                else if (root->data->state.c_array[y] & (BLOCK(1) << x))
                {
                    if (!(root->data->state.empty % 2))   std::cout << "\033[1;34m o \033[0m";
                    else                std::cout << "\033[1;31m o \033[0m";
                }
                else if (!(root->data->state.c_array[y] & (BLOCK(1) << x)))
                {
                    if (root->data->state.empty % 2)      std::cout << "\033[1;34m o \033[0m";
                    else                std::cout << "\033[1;31m o \033[0m";
                }
            }
            std::cout << "|\n   ";
            for (uint16_t i = 0; i < BoardSize; i++)
                std::cout << " ---";
            std::cout << "\n";
        }

        std::cout << "    <";
        for (uint16_t i = 0; i < BoardSize * 2 + 26; i++)
            std::cout << "-";
        std::cout << ">\n";
    }

private:
    static void update(uint64_t ix)
    {

        uint64_t i = log_table_size;
        for (; i < (ix + 1) * BATCH_BoardSize; i++)
            log_table[i] = std::log(i);
        if (i > log_table_size)
            log_table_size = i;
    }

    static void MCTS_master(NODE* root, State* root_state, PREC confidence_bound, bool analytics)
    {
        // Select best child
        NODE* best = nullptr;
        std::list<NODE*> children;
        for (NODE* child : root->children) children.push_back(child);
        uint16_t children_count = root->children.size();
        for (uint16_t i = 0; i < children_count; i++)
        {
            NODE* child = best_child_final(root);
            if (PREC(child->data->visits) / PREC(root->data->visits) > confidence_bound)
            {
                best = child;
                break;
            }
            root->children.remove(child);
        }
        root->children = children;

        if (!best)
        {
            std::cout << "[WARNING]: No action in confidence bound!\n";
            best = best_child_final(root);
        }
        
        if (analytics)
        {
            render_sim_distribution(root);
            render_ucb_distribution(root);
        }

        print_evaluation(best);

        State* best_state = new State(best->data->state);
        root_state->makeMove(best->parent_action);

        delete root;
    }

    static void print_evaluation(NODE* best)
    {
        std::cout << "Action:      " << int32_t(best->parent_action) % BoardSize << "," << int32_t(best->parent_action) / BoardSize << "\n";
        std::cout << "Simulations: " << PREC(int32_t(best->parent->data->visits) / 1000) / 1000 << "M";
        std::cout << " (W:" << int(best->parent->data->results[best->parent->data->state.empty % 2 ? 0 : 1]) << " L:" << int(best->parent->data->results[best->parent->data->state.empty % 2 ? 1 : 0]) << " D:" << int(best->parent->data->results[2]) << ")\n";
        std::cout << "Evaluation:  " << PREC(PREC(best->Q_delta(best->parent->data->state.empty % 2)) / PREC(best->data->visits));
        std::cout << " (W:" << int(best->data->results[best->parent->data->state.empty % 2 ? 0 : 1]) << " L:" << int(best->data->results[best->parent->data->state.empty % 2 ? 1 : 0]) << " D:" << int(best->data->results[2]) << ")\n";
        std::cout << "Confidence:  " << PREC(best->data->visits * 100) / PREC(best->parent->data->visits) << "%\n";
        std::printf("Draw:        %.2f%%\n", PREC(best->data->results[2] * 100) / PREC(best->data->visits));

        std::cout << "    <";
        for (uint16_t i = 0; i < BoardSize * 2 + 26; i++)
            std::cout << "-";
        std::cout << ">\n";
    }

    static NODE* best_child_final(NODE* node)
    {
        NODE* best_child = nullptr;
        PREC result;
        PREC best_result = -100.0;

        // Precompute
        bool turn = node->data->state.empty % 2;

        for (NODE* child : node->children)
        {
            PREC Q_value = PREC(child->Q_delta(turn)) / PREC(child->data->visits);
            result = Q_value;
            if (result > best_result)
            {
                best_result = result;
                best_child = child;
            }
        }
        return best_child;
    }
};

int main()
{
    HOST::init();

    State* state = HOST::create();

    std::cout << state->toString();
    while (!state->isTerminal())
    {
        if (!(state->empty % 2))
            HOST::MCTS_move(state, std::chrono::seconds(1), 0.01, true);
        //HOST::human_move(state);  
        else
            HOST::human_move(state);
        std::cout << state->toString();
    }
}