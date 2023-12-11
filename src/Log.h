#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
*/


#include "Config.h"
#include <iostream>
#include <string>
#include <sstream>

/**
 * @enum LogLevel
 * @brief Enumeration for different levels of logging.
 * @var LogLevel::INFO
 *   Informational level, for general, useful information about the normal operation of the system.
 * @var LogLevel::WARNING
 *   Warning level, for potentially harmful situations.
 * @var LogLevel::ERROR
 *   Error level, for error events that might still allow the application to continue running.
 * @var LogLevel::FATAL
 *   Fatal level, for very severe error events that will presumably lead the application to abort.
 */
enum class LogLevel {
    INFO,
    WARNING,
    ERROR,
    FATAL
};

/**
 * @class Log
 * @brief Class for logging messages.
 * @details This class provides static methods for logging messages at different levels.
*/
class Log {
 public:
    /**
     * @brief Logs a message at the specified level.
     * @param level The level at which to log the message.
     * @param message The message to log.
     */
    static void log(
        LogLevel level,
        const std::string& message);

    /**
     * @brief Logs a message at the specified level with a specified scope.
     * @param level The level at which to log the message.
     * @param message The message to log.
     * @param scope The scope within which to log the message.
     */
    static void log(
        LogLevel level,
        const std::string& message,
        const std::string& scope);

    /**
     * @brief Sets the current log level.
     * @param level The level to set as the current log level.
     */
    static void setLogLevel(LogLevel level);

    /**
     * @brief Returns the current log level.
     * @return The current log level.
     */
    static LogLevel getLogLevel();

    /**
     * @brief Returns the current log level as a string.
     * @return The current log level as a string.
     */
    static std::string getLogLevelStr();

 private:
    // The current log level
    static LogLevel log_level;

    // Prevent instantiation
    Log() = delete;
};
