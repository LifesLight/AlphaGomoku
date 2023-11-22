#pragma once

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

private:
    static LogLevel log_level;
};