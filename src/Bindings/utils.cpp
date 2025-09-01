#include <pybind11/pybind11.h>
#include "utils/json.hpp"
#include "utils/dotenv.hpp"

namespace py = pybind11;

py::object JsonValue::to_py() const {
    switch (type) {
        case Type::OBJECT: {
            py::dict obj;
            for (const auto& [k, v] : std::get<JsonObject>(value)) {
                obj[py::str(k)] = v->to_py();
            }
            return obj;
        }
        case Type::ARRAY: {
            py::list arr;
            for (const auto& v : std::get<JsonArray>(value)) {
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


py::object load_dotenv_py(const std::string& path = ".env") {
    auto env_map = load_dotenv(path);
    py::dict env_dict;
    for (const auto& [k, v] : env_map) {
        env_dict[py::str(k)] = py::str(v);
    }
    return env_dict;
}
py::dict http_get(const std::string& url, const std::map<std::string, std::string>& params = {}, bool use_cache = true) {
    std::string cache_path = "dotenv_cache.txt";
    std::string content;
    if (use_cache) {
        try {
            content = load_cached_dotenv(cache_path);
        } catch (...) {
            content = fetch_and_cache_dotenv(url, cache_path);
        }
    } else {
        content = fetch_and_cache_dotenv(url, cache_path);
    }
    auto json_value = json::parse(content);
    return json_value->to_py().cast<py::dict>();
}

PYBIND11_MODULE(utils, m) {
    m.def("load_dotenv", &load_dotenv_py, py::arg("path") = ".env", "Load environment variables from a .env file");
    m.def("http_get", &http_get, py::arg("url"), py::arg("params") = std::map<std::string, std::string>{}, py::arg("use_cache") = true, "Perform HTTP GET request and return JSON response as dict");
    py::class_<JsonValue>(m, "JsonValue")
        .def("to_py", &JsonValue::to_py, "Convert JsonValue to Python object");
}