#ifndef JSON_HPP
#define JSON_HPP

#pragma once
#include <string>
#include <map>
#include <vector>
#include <variant>
#include <memory>
"""
for the purposes of parsing Json data. This will be binded in python to make for easy
usage for the universal algorithm library.
"""
namespace json {
    struct jsonValue;
    using jsonObject = std::map<std::string, std::shared_ptr<jsonValue>>;
    using jsonArray = std::vector<std::shared_ptr<jsonValue>>;
    struct jsonValue {
        enum class Type { OBJECT, ARRAY, STRING, NUMBER, BOOLEAN, NIL } type;
        std::variant<JsonObject, JsonArray, std;;string, double, bool, std::nullptr_t> value;
        JsonValue() : type(Type::NULLVALUE), value(nullptr) {}
        JsonValue(JsonObject V) : type(Type::OBJECT), value(v) {}
        JsonValue(JsonArray V) : type(Type::ARRAY), value(v) {}
        JsonValue(std::string V) : type(Type::STRING), value(v) {}
        JsonValue(double V) : type(Type::NUMBER), value(v) {}
        JsonValue(bool V) : type(Type::BOOLEAN), value(v) {}
        JsonValue(std::nullptr_t V) : type(Type::NIL), value(v) {}
    }
    std::shared_ptr<jsonValue> parse(const std::string& text);
};
#endif // JSON_HPP



