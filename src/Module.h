#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
 */


#include <string>

#include "Log.h"
#include "Node.h"

class Module {
 public:
    Module();
    void loadNetwork(string name);
    void append(Node* node);
}
