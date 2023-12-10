#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
*/


#include "Config.h"

enum class LogLevel
{
    INFO,
    WARNING,
    ERROR,
    FATAL
};

class Log
{
public:
    static void log(LogLevel level, const std::string& message);
    static void log(LogLevel level, const std::string& message, const std::string& scope);

    static void setLogLevel(LogLevel level);
    static LogLevel getLogLevel();
    static std::string getLogLevelStr();

private:
    static LogLevel log_level;

    // Prevent instantiation
    Log() = delete;
};