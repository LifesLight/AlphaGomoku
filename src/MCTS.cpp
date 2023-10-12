#include "Config.h"
#include "Node.h"
#include "State.h"
#include "Model.h"
#include "Environment.h"


std::string ucb_distribution(Node* root)
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
                        result << std::setw(3) << std::setfill(' ') << int(float(child->meanEvaluation()) * 100);
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

std::string val_distribution(Node* root)
{
    std::ostringstream result;

    result << "\n    <";
    for (uint16_t i = 0; i < BoardSize; i++)
        result << "-";
    result << "     VAL DISTRIBUTION     ";

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
                        result << std::setw(3) << std::setfill(' ') << int(float(-(child->value)) * 100);
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

std::string sim_distribution(Node* root)
{
    std::ostringstream result;

    result << "\n    <";
    for (uint16_t i = 0; i < BoardSize; i++)
        result << "-";
    result << " SIMULATIONS DISTRIBUTION ";

    for (uint16_t i = 0; i < BoardSize; i++)
        result << "-";
    result << ">\n   ";

    float max_visits = 0;
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
                        result << std::setw(3) << std::setfill(' ') << int(float(child->visits) / max_visits * 100);
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

int main(int argc, const char* argv[])
{
    if (argc == 1)
    {
        std::cout << "[FATAL]: Missing model path" << std::endl;
        return 0;
    }

    int simulations = 100;
    if (argc == 3)
        simulations = std::stoi(argv[2]);

    Model* neural_network = new Model(argv[1], torch::kMPS, "Testmodel");

    Environment* env = new Environment(nullptr, neural_network);

    int turn = 0;

    while (!env->isFinished())
    {
        if (turn % 2)
        {
            uint16_t computedMove = env->calculateNextMove(simulations);
            env->makeMove(computedMove);
            Node* cn = env->getNode(!env->getNextColor());
            std::cout << sim_distribution(cn->parent) << std::endl;
            std::cout << ucb_distribution(cn->parent) << std::endl;
            std::cout << val_distribution(cn->parent) << std::endl;
            std::cout << Environment::nodeAnalytics(cn) << std::endl;
        }
        else
        {
            std::string x_string, y_string;
            uint8_t x, y;
            std::cout << "Move:" << std::endl << "X:";
            std::cin >> x_string;
            std::cout << "Y:";
            std::cin >> y_string;
            x = std::stoi(x_string);
            y = std::stoi(y_string);
            env->makeMove(x, y);
        }
        std::cout << env->toString() << std::endl;
        turn++;
    }
    
    return 1;
}
