#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <mutex>

enum class LogLevel { INFO = 0, WARNING = 1, ERROR = 2 };

class Logger {
public:
    // Set minimum log level for output
    static void setLogLevel(LogLevel level) {
        getMutex().lock();
        getLogLevel() = level;
        getMutex().unlock();
    }

    // Log message to console with optional tag and timestamp
    static void log(const std::string& msg, LogLevel level = LogLevel::INFO, const std::string& tag = "", bool withTimestamp = true) {
        if (level < getLogLevel()) return;

        std::lock_guard<std::mutex> lock(getMutex());
        if (withTimestamp) {
            std::cout << "[" << currentDateTime() << "] ";
        }
        std::cout << "[" << toString(level) << "]";
        if (!tag.empty()) {
            std::cout << "[" << tag << "]";
        }
        std::cout << " " << msg;
        if (level == LogLevel::ERROR) {
            std::cerr << std::endl;
        } else {
            std::cout << std::endl;
        }
    }

    // Log message to file with timestamp and tag
    static void logToFile(const std::string& msg, const std::string& path, LogLevel level = LogLevel::INFO, const std::string& tag = "") {
        if (level < getLogLevel()) return;

        std::lock_guard<std::mutex> lock(getMutex());
        std::ofstream logfile(path, std::ios_base::app);
        if (!logfile.is_open()) {
            // Optional: fallback to console error
            std::cerr << "[ERROR][Logger] Failed to open log file: " << path << std::endl;
            return;
        }
        logfile << "[" << currentDateTime() << "] "
                << "[" << toString(level) << "]";
        if (!tag.empty()) {
            logfile << "[" << tag << "]";
        }
        logfile << " " << msg << std::endl;
    }

private:
    // Current minimum log level, default INFO
    static LogLevel& getLogLevel() {
        static LogLevel level = LogLevel::INFO;
        return level;
    }

    // Mutex for thread safety
    static std::mutex& getMutex() {
        static std::mutex mtx;
        return mtx;
    }

    // Get string representation of LogLevel
    static std::string toString(LogLevel level) {
        switch (level) {
            case LogLevel::INFO:    return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR:   return "ERROR";
            default:               return "UNKNOWN";
        }
    }

    // Get current datetime string "YYYY-MM-DD HH:MM:SS"
    static std::string currentDateTime() {
        std::time_t now = std::time(nullptr);
        char buf[20];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        return std::string(buf);
    }
};

#endif // LOGGER_HPP