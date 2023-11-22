#include "Log.h"

LogLevel Log::log_level = LogLevel::WARNING;

void Log::setLogLevel(LogLevel level)
{
    log_level = level;
}

LogLevel Log::getLogLevel()
{
    return log_level;
}

void Log::log(LogLevel level, const std::string& message)
{
    log(level, message, "");
}

void Log::log(LogLevel level, const std::string& message, const std::string& scope)
{
    if (level >= log_level)
    {
        std::stringstream output;

        std::string level_string;
        switch (level)
        {
        case LogLevel::INFO:
            level_string = "INFO";
            break;
        case LogLevel::WARNING:
            level_string = "WARNING";
            break;
        case LogLevel::ERROR:
            level_string = "ERROR";
            break;
        case LogLevel::FATAL:
            level_string = "FATAL";
            break;
        }

        if (level == LogLevel::FATAL || level == LogLevel::ERROR)
            output << "\033[1;31m";
        else if (level == LogLevel::WARNING)
            output << "\033[1;33m";

        output << "[" << level_string << "]";
        if (scope != "")
            output << "[" << scope << "]";

        output << ": " << message << "\033[0m" << std::endl;

        if (level == LogLevel::FATAL || level == LogLevel::ERROR)
            output << std::flush;

        std::cout << output.str();
    }
}