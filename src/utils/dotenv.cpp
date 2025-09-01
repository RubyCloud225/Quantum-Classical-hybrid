#include <fstream>
#include <sstream>
#include <filesystem>
#include <iomanip>
#include <openssl/sha.h>
#include <string>
#include "json_util.hpp"
#include "dotenv.hpp"
#include <cstdlib>
#include <curl/curl.h>
#include <unordered_map>

namespace fs = std::filesystem;

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Cache to store fetched .env files
static std::unordered_map<std::string, std::string> dotenv_cache;
static std::string cache_folder = "cache";

std::string sha1(const std::string& str) {
    unsigned char hash[SHA_DIGEST_LENGTH];
    SHA1(reinterpret_cast<const unsigned char*>(str.c_str()), str.size(), hash);
    std::ostringstream oss;
    for (int i = 0; i < SHA_DIGEST_LENGTH; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return oss.str();
}

// Cache folder
void set_cache_folder(const std::string& folder) {
    cache_folder = folder;
    if (!fs::exists(cache_folder)) {
        fs::create_directories(cache_file_path);
    }
}
// Load and save cache
void save_cache() {
    std::ofstream ofs(cache_folder + "/dotenv_cache.txt", std::ios::binary);
    if (ofs.is_open()) {
        for (const auto& [key, value] : dotenv_cache) {
            ofs << key << "\n" << value << "\n===END===\n";
        }
        ofs.close();
    }
}
// Function to cache the fetched .env content
void load_cache(const std::string& cache_filename = "dotenv_cache.txt") {
    std::string path = cache_folder + "/" + cache_filename;
    if (!fs::exists(path)) return;
    std::ifstream in(path);
    if (in.is_open()) {
        std::string line;
        std::string key, value;
        while (std::getline(in, line)) {
            if (line == "===END===") {
                dotenv_cache[key] = value;
                key.clear();
                value.clear();
            } else if (key.empty()) {
                key = line;
            } else {
                if (!value.empty()) value += "\n";
                value += line;
            }
        }
        in.close();
    }
}
// Clear cache
void clearcache(const std::string& cache_filename = "dotenv_cache.txt") {
    std::string path = cache_folder + "/" + cache_filename;
    if (fs::exists(path)) {
        fs::remove(path);
    }
    dotenv_cache.clear();
}
// fetch and cache .env file from URL
std::string fetch_and_cache_dotenv(const std::string& url, const std::string& cache_filepath = "dotenv_cache.txt") {
    CURL* curl;
    CURLcode res;
    curl = curl_easy_init();
    std::string readBuffer;
    if (!curl) throw std::runtime_error("Could not initialize curl");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("curl_easy_perform() failed: " + std::string(curl_easy_strerror(res)));
    }
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("curl_easy_perform() failed: " + std::string(curl_easy_strerror(res)));
    }
    std::ofstream cache_file(cache_folder + "/" + cache_filepath);
    if (cache_file.is_open()) {
        cache_file << readBuffer;
        cache_file.close();
    }
    return readBuffer;
}

std::string load_cached_dotenv(const std::string& cache_filepath = "dotenv_cache.txt") {
    std::string path = cache_folder + "/" + cache_filepath;
    if (!fs::exists(path)) {
        throw std::runtime_error("Cache file does not exist");
    }
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open cache file");
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}
// Https:// or http:// URL support
std::string http_get(const std::string& url, const std::map<std::string, std::string>& params = {}, bool use_cache = true) {
    std::ostringstream key_stream;
    key_stream << url;
    for (auto& [k, v] : params) {
        key_stream << "|" << k << "=" << v;
    }
    std::string cache_key = sha1(key_stream.str()) + ".cache";
    std::string cache_file = cache_key + ".json";
    // Check cache
    std::string cache_path = cache_folder + "/" + cache_file;
    if (fs::exists(cache_path) && use_cache) {
        std::ifstream in(cache_path);
        std::stringstream buffer;
        buffer << in.rdbuf();
        return buffer.str();
    }
    // build full URL with params
    std::string full_url = url;
    if (!params.empty()) {
        full_url += "?";
        bool first = true;
        for (const auto& [k, v] : params) {
            if (!first) full_url += "&";
            first = false;
            char* encoded_key = curl_easy_escape(nullptr, k.c_str(), k.size());
            full_url += k + "=" + std::string(encoded_key);
            curl_free(encoded_key);
        }
    }
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("Could not initialize curl");
    std::string readBuffer;
    curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("curl_easy_perform() failed: " + std::string(curl_easy_strerror(res)));
    }
    curl_easy_cleanup(curl);
    // Save to cache
    std::ofstream out(cache_path);
    if (out.is_open()) {
        out << readBuffer;
        out.close();
    }
    return readBuffer;
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