#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
*/


#include "Config.h"
#include "Utilities.h"
#include "Datapoint.h"

/*
Interface to load and store datapoints
Assumes that database is without formating errors, no checks included for performance
*/

class Storage
{
public:
    Storage(std::string path);

    // Writes all changes to file
    // This interface becomes unusable after calling this function
    void applyChanges();
    // How many datapoints are in database
    int getDatapointCount();
    // Delete and store datapoints
    void deleteDatapoint(int index);
    void storeDatapoint(Datapoint data);
    // Retrieve datapoint
    Datapoint getDatapoint(int index);
    // Reduce number of datapoints to value, deletes oldest
    void constrain(int max_datapoints);

private:
    Datapoint lineToDatapoint(std::string line);
    std::string datapointToLine(Datapoint data);

    std::string file_path;
    std::vector<std::string> lines;
    bool change_made;
};