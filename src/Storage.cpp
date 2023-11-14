#include "Storage.h"

Storage::Storage(std::string path)
    : file_path(path), change_made(false)
{
    if (std::filesystem::exists(path))
    {
        if (Utils::checkEnv("LOGGING", "INFO"))
            std::cout << "[Storage][I]: Created interface to existing database: " << path << std::endl;
        // We assume the file is already populated with the max amount of datapoints
        lines.reserve(MaxDatapoints);

        std::ifstream reader(path);
        if (!reader)
        {
            ForcePrint("[Storage][E]: Failed to open file for reading");
            return;
        }

        // Read lines
        std::string line;
        while (std::getline(reader, line))
            lines.push_back(line);
        reader.close();
    }
    else
    {
        if (Utils::checkEnv("LOGGING", "INFO"))
            std::cout << "[Storage][I]: Created interface to new database: " << path << std::endl;
    }
}

void Storage::applyChanges()
{
    if (!change_made)
    {
        if (Utils::checkEnv("LOGGING", "INFO"))
            std::cout << "[Storage][I]: Skipped applying changes to file (no changes)" << std::endl;
        return;
    }

    std::ofstream output(file_path, std::ios::trunc);

    if (!output) {
        std::cout << "[Storage][E]: Failed to open file for writing" << std::endl;
        return;
    }

    for (const std::string &line : lines) {
        output << line << '\n';
    }

    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Storage][I]: Applied changes to file" << std::endl;

    output.close();
}

Datapoint Storage::lineToDatapoint(std::string line)
{
    Datapoint data;
    int index = 0;

    // Read moves
    while (line[std::max(0, index - 1)] != ';')
    {
        std::string move;
        while (line[index] != ',' && line[index] != ';')
        {
            move.push_back(line[index++]);
        }

        data.moves.push_back(std::stoi(move));
        index++;
    }

    // Read best move
    std::string move;
    while(line[index] != ';')
    {
        move.push_back(line[index++]);
    }
    index++;
    data.best_move = std::stoi(move);

    // Read outcome
    std::string outcome = "";
    outcome += line[index];
    data.winner = std::stoi(outcome);

    return data;
}

std::string Storage::datapointToLine(Datapoint data)
{
    std::string line;
    for (index_t move : data.moves)
    {
        line += std::to_string(move);
        if (move != data.moves.back())
            line += ',';
    }
    line += ';';
    line += std::to_string(data.best_move);
    line += ';';
    line += std::to_string(data.winner);
    return line;
}

int Storage::getDatapointCount()
{
    return lines.size();
}

void Storage::deleteDatapoint(int id)
{
    if (id < getDatapointCount())
    {
        change_made = true;
        lines.erase(lines.begin() + id);
    }
    else
        ForcePrint("[Storage][W]: Tried to delete non existent datapoint");
}

void Storage::storeDatapoint(Datapoint data)
{
    change_made = true;
    lines.push_back(datapointToLine(data));
}

Datapoint Storage::getDatapoint(int id)
{
    if (id >= getDatapointCount())
    {
        ForcePrint("[Storage][W]: Tried to get non existent datapoint");
        return Datapoint();
    }

    return lineToDatapoint(lines[id]);
}

void Storage::constrain(int count)
{
    if (getDatapointCount() < count)
        return;
    
    int lowest_index = getDatapointCount() - count;
    lines.erase(lines.begin(), lines.begin() + lowest_index);

    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Storage][I]: Deleted " << lowest_index << " datapoints" << std::endl;
    
    change_made = true;
}