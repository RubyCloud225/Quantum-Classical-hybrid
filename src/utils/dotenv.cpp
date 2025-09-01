#include <fstream>
#include <string>
#include "json_util.hpp"
#include "dotenv.hpp"
#include <cstdlib>
#include <curl/curl.h>
#include <unordered_map>

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Function to cache the fetched .env content
std::string fetch_and_cache_dotenv(const std::string& url, const std::string& cache_path) {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        if(res != CURLE_OK) {
            throw std::runtime_error("Failed to fetch .env file from URL");
        }

        // Cache the content
        std::ofstream cache_file(cache_path);
        if (cache_file.is_open()) {
            cache_file << readBuffer;
            cache_file.close();
        } else {
            throw std::runtime_error("Failed to write cache file");
        }
    } else {
        throw std::runtime_error("Failed to initialize CURL");
    }

    return readBuffer;
}
std::string load_cached_dotenv(const std::string& cache_path) {
    std::ifstream cache_file(cache_path);
    if (cache_file.is_open()) {
        std::string content((std::istreambuf_iterator<char>(cache_file)), std::istreambuf_iterator<char>());
        cache_file.close();
        return content;
    } else {
        throw std::runtime_error("Failed to open cache file");
    }
}
std::string make_cache_key(const std::string& url, const std::map<std::string, std::string>& params) {
    std::string key = url;
    for (const auto& [k, v] : params) {
        key += "_" + k + "=" + v;
    }
    return key;
}
std::map<std::string, std::string> load_dotenv(const std::string& path) {
    std::map<std::string, std::string> env_map;
    std::string content;

    // Check if path is a URL
    if (path.rfind("http://", 0) == 0 || path.rfind("https://", 0) == 0) {
        // Create a cache key based on the URL
        std::string cache_key = "dotenv_cache.txt"; // Simplified for this example
        try {
            content = load_cached_dotenv(cache_key);
        } catch (...) {
            content = fetch_and_cache_dotenv(path, cache_key);
        }
    } else {
        // Load from local file
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open .env file");
        }
        content.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
    }

    std::istringstream ss(content);
    std::string line;
    while (std::getline(ss, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            env_map[key] = value;
            // Optionally set in environment variables
            setenv(key.c_str(), value.c_str(), 1);
        }
    }

    return env_map;
}

void clear_cache() {
    std::string cache_path = "dotenv_cache.txt";
    if (std::remove(cache_path.c_str()) != 0) {
        throw std::runtime_error("Failed to clear cache");
    }
}