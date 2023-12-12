/**
 * Copyright (c) Alexander Kurtz 2023
 */

#include <iostream>

#include "State.h"
#include "Log.h"
#include "Types.h"

using std::cout;
using std::endl;

i32_t main(i32_t argc, char const* argv[]) {
    Log::setLogLevel(LogLevel::INFO);

    State s = State();
    s.makeMove(0, 0);
    s.makeMove(0, 1);
    s.makeMove(0, 2);
    cout << s.str() << endl;
    return 0;
}
