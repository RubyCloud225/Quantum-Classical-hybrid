#ifndef JSON_HPP
#define JSON_HPP

#pragma once
#include <string>
#include <map>
#include <vector>
#include <variant>
#include <memory>

namespace json {
    struct jsonValue;
    using jsonObject = std::map<std::string, std::shared_ptr<jsonValue>>;
    using jsonArray = std::vector<std::shared_ptr<jsonValue>>;
    struct jsonValue {
        enum class Type { OBJECT, ARRAY, STRING, NUMBER, BOOLEAN, NIL } type;
        std::variant<jsonObject, jsonArray, std::string, double, bool, std::nullptr_t> value;

        jsonValue() : type(Type::NIL), value(nullptr) {}
        jsonValue(jsonObject v) : type(Type::OBJECT), value(v) {}
        jsonValue(jsonArray v) : type(Type::ARRAY), value(v) {}
        jsonValue(std::string v) : type(Type::STRING), value(v) {}
        jsonValue(double v) : type(Type::NUMBER), value(v) {}
        jsonValue(bool v) : type(Type::BOOLEAN), value(v) {}
        jsonValue(std::nullptr_t v) : type(Type::NIL), value(v) {}
    };
    std::shared_ptr<jsonValue> parse(const std::string& text);
};
#endif // JSON_HPP



