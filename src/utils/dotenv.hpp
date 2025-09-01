#pragma once
#include "json.hpp"
#include <map>
#include <string>

std::string load_cached_dotenv(const std::string& cache_filepath = "dotenv_cache.txt");
std::string fetch_and_cache_dotenv(const std::string& url, const std::string& cache_filepath = "dotenv_cache.txt");
std::string http_get(const std::string& url, const std::map<std::string, std::string>& params = {}, bool use_cache = true);
std::map<std::string, std::string> load_dotenv(const std::string& path = ".env");
