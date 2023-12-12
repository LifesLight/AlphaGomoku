/**
 * Copyright (c) Alexander Kurtz 2023
*/


#include "Log.h"

LogLevel Log::log_level = LogLevel::WARNING;

void Log::setLogLevel(LogLevel level) {
    log_level = level;
}

LogLevel Log::getLogLevel() {
    return log_level;
}

string Log::getLogLevelStr() {
    switch (log_level) {
    case LogLevel::INFO:
        return "INFO";
    case LogLevel::WARNING:
        return "WARNING";
    case LogLevel::ERROR:
        return "ERROR";
    case LogLevel::FATAL:
        return "FATAL";
    }

    return "ERR";
}

void Log::log(LogLevel level, const string& message) {
    log(level, message, "");
}

void Log::log(
    LogLevel level,
    const string& message,
    const string& scope) {

    if (level >= log_level) {
        stringstream output;

        string level_string;
        switch (level) {
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

        output << ": " << message << "\033[0m" << endl;

        if (level == LogLevel::FATAL || level == LogLevel::ERROR)
            output << flush;

        cout << output.str();
    }
}
