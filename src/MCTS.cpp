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
        std::cout << "\n";
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

static std::string sim_distribution(Node* root)
{
    std::ostringstream result;

    result << "\n    <";
    for (uint16_t i = 0; i < BoardSize; i++)
        result << "-";
    result << " SIMULATIONS DISTRIBUTION ";

    for (uint16_t i = 0; i < BoardSize; i++)
        result << "-";
    result << ">\n   ";

    FloatPrecision max_visits = 0;
    for (Node* child : root->children)
        if (child->visits > max_visits)
            max_visits = child->visits;

    for (uint16_t i = 0; i < BoardSize; i++)
        result << " ---";
    result << "\n";
    for (int16_t y = BoardSize - 1; y >= 0; y--)
    {
        result << std::setw(3) << std::setfill(' ') << y;
        for (uint16_t x = 0; x < BoardSize; x++)
        {
            result << "|";
            if (!(root->state->m_array[y] & (BLOCK(1) << x)))
            {
                for (Node* child : root->children)
                    if (child->parent_action == (y * BoardSize + x))
                        result << std::setw(3) << std::setfill(' ') << int(FloatPrecision(child->visits) / max_visits * 100);
            }
            else if (root->state->c_array[y] & (BLOCK(1) << x))
            {
                if (root->state->nextColor())
                    result << "\033[1;34m o \033[0m";
                else
                    result << "\033[1;31m o \033[0m";
            }
            else if (!(root->state->c_array[y] & (BLOCK(1) << x)))
            {
                if (!root->state->nextColor())
                    result << "\033[1;34m o \033[0m";
                else
                    result << "\033[1;31m o \033[0m";
            }
        }
        result << "|\n   ";
        for (uint16_t i = 0; i < BoardSize; i++)
            result << " ---";
        result << "\n";
    }

    result << "  ";
    for (uint16_t i = 0; i < BoardSize; i++)
        result << " " << std::setw(3) << std::setfill(' ') << i;

    result << "\n    <";
    for (uint16_t i = 0; i < BoardSize * 2 + 26; i++)
        result << "-";
    result << ">\n";

    return result.str();
}


static std::string ucb_distribution(Node* root)
{
    std::ostringstream result;

    result << "\n    <";
    for (uint16_t i = 0; i < BoardSize; i++)
        result << "-";
    result << "     UCB DISTRIBUTION     ";

    for (uint16_t i = 0; i < BoardSize; i++)
        result << "-";
    result << ">";

    result << "\n   ";
    for (uint16_t i = 0; i < BoardSize; i++)
        result << " ---";
    result << "\n";
    for (int16_t y = BoardSize - 1; y >= 0; y--)
    {
        result << std::setw(3) << std::setfill(' ') << y;
        for (uint16_t x = 0; x < BoardSize; x++)
        {
            result << "|";
            if (!(root->state->m_array[y] & (BLOCK(1) << x)))
            {
                for (Node* child : root->children)
                    if (child->parent_action == (y * BoardSize + x))
                        result << std::setw(3) << std::setfill(' ') << int(FloatPrecision(child->qDelta(!root->state->nextColor())) / FloatPrecision(child->visits) * 100);
            }
            else if (root->state->c_array[y] & (BLOCK(1) << x))
            {
                if (root->state->nextColor())
                    result << "\033[1;34m o \033[0m";
                else
                    result << "\033[1;31m o \033[0m";
            }
            else if (!(root->state->c_array[y] & (BLOCK(1) << x)))
            {
                if (!root->state->nextColor())
                    result << "\033[1;34m o \033[0m";
                else
                    result << "\033[1;31m o \033[0m";
            }
        }
        result << "|\n   ";
        for (uint16_t i = 0; i < BoardSize; i++)
            result << " ---";
        result << "\n";
    }

    result << "  ";
    for (uint16_t i = 0; i < BoardSize; i++)
        result << " " << std::setw(3) << std::setfill(' ') << i;;
    result << "\n    <";
    for (uint16_t i = 0; i < BoardSize * 2 + 26; i++)
        result << "-";
    result << ">\n";

    return result.str();
}


private:
    static void MCTS_master(Node* root, State* root_state, FloatPrecision confidence_bound, bool analytics)
    {
        // Select best child
        Node* best = nullptr;
        best = root->absBestChild(confidence_bound);
        bool confidenceBoundFailed = false;

        if (!best)
        {
            confidenceBoundFailed = true;
            best = root->absBestChild();
        }
        
        if (analytics)
        {
            std::cout << sim_distribution(root);
            std::cout << ucb_distribution(root);
        }

        if (confidenceBoundFailed)
            std::cout << "[WARNING]: No action in confidence bound!\n";

        std::cout << evaluation(best);

        State* best_state = new State(best->state);
        root_state->makeMove(best->parent_action);

        delete root;
    }

    static std::string evaluation(Node* best)
    {
        std::ostringstream result;
        result << "Action:      " << int32_t(best->parent_action) % BoardSize << "," << int32_t(best->parent_action) / BoardSize << "\n";
        result << "Simulations: " << FloatPrecision(int32_t(best->parent->visits) / 1000) / 1000 << "M";
        result << " (W:" << int(best->parent->results[best->parent->state->nextColor() ? 1 : 0]) << " L:" << int(best->parent->results[best->parent->state->nextColor() ? 0 : 1]) << " D:" << int(best->parent->results[2]) << ")\n";
        result << "Evaluation:  " << FloatPrecision(FloatPrecision(best->qDelta(!best->parent->state->nextColor())) / FloatPrecision(best->visits));
        result << " (W:" << int(best->results[best->parent->state->nextColor() ? 1 : 0]) << " L:" << int(best->results[best->parent->state->nextColor() ? 0 : 1]) << " D:" << int(best->results[2]) << ")\n";
        result << "Confidence:  " << FloatPrecision(best->visits * 100) / FloatPrecision(best->parent->visits) << "%\n";
        result << "Draw:        " << std::fixed << std::setprecision(2) << FloatPrecision(best->results[2] * 100) / FloatPrecision(best->visits) << "%\n";
        result << "    <";
        for (uint16_t i = 0; i < BoardSize * 2 + 26; i++)
            result << "-";
        result << ">\n";

        return result.str();
    }
};

int main()
{
    HOST::init();

    State* state = HOST::create();

    std::cout << state->toString();
    while (!state->isTerminal())
    {
        if (state->nextColor())
            HOST::MCTS_move(state, std::chrono::seconds(10), 0.01, true);
        else
            HOST::human_move(state);
        std::cout << state->toString();
    }
}