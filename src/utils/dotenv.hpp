#pragma once
#include "json.hpp"
#include <map>
#include <string>


std::map<std::string, std::string> load_dotenv(const std::string& path = ".env");