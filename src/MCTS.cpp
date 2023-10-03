#include "Config.h"
#include "Node.h"
#include "State.h"

class HOST
{
public:

    static void init()
    {
        for (uint32_t i = 1; i < MaxSimulations; i++)
            logTable[i] = std::log(i);
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

    static void MCTS_move(State* root_state, std::chrono::milliseconds time, FloatPrecision confidence_bound, bool analytics)
    {
        Node* root = new Node(root_state);
        // Build Tree
        uint64_t i;
        auto start = std::chrono::high_resolution_clock::now();
        while (std::chrono::high_resolution_clock::now() - start < time)
        {
            for (i = 0; i < 1000; i++)
            {
                Node* node = root->policy();
                node->rollout();
            }
        }
        MCTS_master(root, root_state, confidence_bound, analytics);
    }

    static void MCTS_move(State* root_state, uint64_t simulations, FloatPrecision confidence_bound, bool analytics)
    {
        Node* root = new Node(root_state);
        // Build Tree
        for (uint64_t i = 0; i < simulations; i++)
        {
            Node* node = root->policy();
            node->rollout();
        }
        MCTS_master(root, root_state, confidence_bound, analytics);
    }

    static void render_sim_distribution(Node* root)
    {
        std::cout << "\n    <";
        for (uint16_t i = 0; i < BoardSize; i++)
            std::cout << "-";
        std::cout << " SIMULATIONS DISTRIBUTION ";

        for (uint16_t i = 0; i < BoardSize; i++)
            std::cout << "-";
        std::cout << ">\n";

        FloatPrecision max_visits = 0;
        for (Node* child : root->children)
            if (child->visits > max_visits)
                max_visits = child->visits;

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
                if (!(root->state.m_array[y] & (BLOCK(1) << x)))
                {
                    for (Node* child : root->children)
                        if (child->parent_action == (y * BoardSize + x))
                            std::printf("%3d", int(FloatPrecision(child->visits) / max_visits * 100));
                }

                else if (root->state.c_array[y] & (BLOCK(1) << x))
                {
                    if (!(root->state.empty % 2))   std::cout << "\033[1;34m o \033[0m";
                    else                std::cout << "\033[1;31m o \033[0m";
                }
                else if (!(root->state.c_array[y] & (BLOCK(1) << x)))
                {
                    if (root->state.empty % 2)      std::cout << "\033[1;34m o \033[0m";
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

    static void render_ucb_distribution(Node* root)
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
                if (!(root->state.m_array[y] & (BLOCK(1) << x)))
                {
                    for (Node* child : root->children)
                        if (child->parent_action == (y * BoardSize + x))
                            std::printf("%+3d", int(FloatPrecision(child->qDelta(root->state.empty % 2)) / FloatPrecision(child->visits) * 100));
                }

                else if (root->state.c_array[y] & (BLOCK(1) << x))
                {
                    if (!(root->state.empty % 2))   std::cout << "\033[1;34m o \033[0m";
                    else                std::cout << "\033[1;31m o \033[0m";
                }
                else if (!(root->state.c_array[y] & (BLOCK(1) << x)))
                {
                    if (root->state.empty % 2)      std::cout << "\033[1;34m o \033[0m";
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
    static void MCTS_master(Node* root, State* root_state, FloatPrecision confidence_bound, bool analytics)
    {
        // Select best child
        Node* best = nullptr;
        std::list<Node*> children;
        for (Node* child : root->children) children.push_back(child);
        uint16_t children_count = root->children.size();
        for (uint16_t i = 0; i < children_count; i++)
        {
            Node* child = best_child_final(root);
            if (FloatPrecision(child->visits) / FloatPrecision(root->visits) > confidence_bound)
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

        State* best_state = new State(best->state);
        root_state->makeMove(best->parent_action);

        delete root;
    }

    static void print_evaluation(Node* best)
    {
        std::cout << "Action:      " << int32_t(best->parent_action) % BoardSize << "," << int32_t(best->parent_action) / BoardSize << "\n";
        std::cout << "Simulations: " << FloatPrecision(int32_t(best->parent->visits) / 1000) / 1000 << "M";
        std::cout << " (W:" << int(best->parent->results[best->parent->state.empty % 2 ? 0 : 1]) << " L:" << int(best->parent->results[best->parent->state.empty % 2 ? 1 : 0]) << " D:" << int(best->parent->results[2]) << ")\n";
        std::cout << "Evaluation:  " << FloatPrecision(FloatPrecision(best->qDelta(best->parent->state.empty % 2)) / FloatPrecision(best->visits));
        std::cout << " (W:" << int(best->results[best->parent->state.empty % 2 ? 0 : 1]) << " L:" << int(best->results[best->parent->state.empty % 2 ? 1 : 0]) << " D:" << int(best->results[2]) << ")\n";
        std::cout << "Confidence:  " << FloatPrecision(best->visits * 100) / FloatPrecision(best->parent->visits) << "%\n";
        std::printf("Draw:        %.2f%%\n", FloatPrecision(best->results[2] * 100) / FloatPrecision(best->visits));

        std::cout << "    <";
        for (uint16_t i = 0; i < BoardSize * 2 + 26; i++)
            std::cout << "-";
        std::cout << ">\n";
    }

    static Node* best_child_final(Node* node)
    {
        Node* best_child = nullptr;
        FloatPrecision result;
        FloatPrecision best_result = -100.0;

        // Precompute
        bool turn = node->state.empty % 2;

        for (Node* child : node->children)
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
};

int main()
{
    HOST::init();

    State* state = HOST::create();

    std::cout << state->toString();
    while (!state->isTerminal())
    {
        if (!(state->empty % 2))
            HOST::MCTS_move(state, 1000000, 0.01, true);
        //HOST::human_move(state);  
        else
            HOST::human_move(state);
        std::cout << state->toString();
    }
}