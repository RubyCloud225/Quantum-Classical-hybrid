#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <pybind11/pybind11.h>
#include "utils/json.hpp"
#include "utils/dotenv.hpp"

namespace py = pybind11;

py::object json::jsonValue::to_py() const {
    switch (type) {
        case Type::OBJECT: {
            py::dict obj;
            for (const auto& [k, v] : std::get<json::jsonObject>(value)) {
                obj[py::str(k)] = v->to_py();
            }
            return obj;
        }
        case Type::ARRAY: {
            py::list arr;
            for (const auto& v : std::get<json::jsonArray>(value)) {
                arr.append(v->to_py());
            }
            return arr;
        }
        case Type::STRING:
            return py::str(std::get<std::string>(value));
        case Type::NUMBER:
            return py::float_(std::get<double>(value));
        case Type::BOOLEAN:
            return py::bool_(std::get<bool>(value));
        case Type::NIL:
            return py::none();
        default:
            throw std::runtime_error("Unknown JSON value type");
    }
}



// Python wrapper for load_dotenv with caching
py::dict load_dotenv_py(const std::string& path = ".env", bool use_cache = true) {
    std::string content;
    std::string cache_path = "dotenv_cache.txt";

    if (use_cache) {
        try {
            content = load_cached_dotenv(cache_path);
        } catch (...) {
            content = fetch_and_cache_dotenv(path, cache_path);
        }
    } else {
        content = fetch_and_cache_dotenv(path, cache_path);
    }

    auto env_map = load_dotenv(path);
    py::dict env_dict;
    for (const auto& [k, v] : env_map) {
        env_dict[py::str(k)] = py::str(v);
    }
    return env_dict;
}