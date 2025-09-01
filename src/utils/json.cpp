#include "json.hpp"
#include <sstream>
#include <cctype>
#include <stdexcept>

namespace json{
    std::shared_ptr<JsonValue> parse(std::istringstream& ss);
    void skipWhitespace(std::istringstream& ss){
        while (ss && std::isspace(ss.peek())) ss.get();
    }
    std::shared_ptr<JsonValue> parseString(std::istringstream& ss){
        std::string result;
        if (ss.get() != '"') throw std::runtime_error("Expected '\"' at the beginning of string");
        while (ss && ss.peek() != '"'){
            if (ss.peek() == '\\'){
                ss.get();
                char escaped = ss.get();
                switch (escaped){
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    default: throw std::runtime_error("Invalid escape sequence");
                }
            } else {
                result += ss.get();
            }
        }
        if (ss.get() != '"') throw std::runtime_error("Expected '\"' at the end of string");
        return std::make_shared<JsonValue>(result);
    }
    std::shared_ptr<JsonValue> parseNumber(std::istringstream& ss){
        std::string numStr;
        if (ss.peek() == '-') numStr += ss.get();
        while (ss && std::isdigit(ss.peek())) numStr += ss.get();
        if (ss.peek() == '.'){
            numStr += ss.get();
            while (ss && std::isdigit(ss.peek())) numStr += ss.get();
        }
        if (ss.peek() == 'e' || ss.peek() == 'E'){
            numStr += ss.get();
            if (ss.peek() == '+' || ss.peek() == '-') numStr += ss.get();
            while (ss && std::isdigit(ss.peek())) numStr += ss.get();
        }
        return std::make_shared<JsonValue>(std::stod(numStr));
    }
    std::shared_ptr<JsonValue> parseLiteral(std::istringstream& ss){
        if (ss.str().compare(ss.tellg(), 4, "true") == 0){
            ss.seekg(ss.tellg() + std::streamoff(4));
            return std::make_shared<JsonValue>(true);
        } else if (ss.str().compare(ss.tellg(), 5, "false") == 0){
            ss.seekg(ss.tellg() + std::streamoff(5));
            return std::make_shared<JsonValue>(false);
        } else if (ss.str().compare(ss.tellg(), 4, "null") == 0){
            ss.seekg(ss.tellg() + std::streamoff(4));
            return std::make_shared<JsonValue>(nullptr);
        }
        throw std::runtime_error("Invalid literal");
    }
    std::shared_ptr<JsonValue> parseArray(std::istringstream& ss){
        if (ss.get() != '[') throw std::runtime_error("Expected '[' at the beginning of array");
        JsonArray arr;
        skipWhitespace(ss);
        if (ss.peek() == ']'){
            ss.get();
            return std::make_shared<JsonValue>(arr);
        }
        while (ss){
            skipWhitespace(ss);
            arr.push_back(parse(ss));
            skipWhitespace(ss);
            if (ss.peek() == ','){
                ss.get();
                continue;
            } else if (ss.peek() == ']'){
                ss.get();
                break;
            } else {
                throw std::runtime_error("Expected ',' or ']' in array");
            }
        }
        return std::make_shared<JsonValue>(arr);
    }
    std::shared_ptr<JsonValue> parseObject(std::istringstream& ss){
        if (ss.get() != '{') throw std::runtime_error("Expected '{' at the beginning of object");
        JsonObject obj;
        skipWhitespace(ss);
        if (ss.peek() == '}'){
            ss.get();
            return std::make_shared<JsonValue>(obj);
        }
        while (ss){
            skipWhitespace(ss);
            auto key = parseString(ss);
            skipWhitespace(ss);
            if (ss.get() != ':') throw std::runtime_error("Expected ':' after key in object");
            skipWhitespace(ss);
            obj[std::get<std::string>(key->value)] = parse(ss);
            skipWhitespace(ss);
            if (ss.peek() == ','){
                ss.get();
                continue;
            } else if (ss.peek() == '}'){
                ss.get();
                break;
            } else {
                throw std::runtime_error("Expected ',' or '}' in object");
            }
        }
        return std::make_shared<JsonValue>(obj);
    }
    std::shared_ptr<JsonValue> parse(std::istringstream& ss){
        skipWhitespace(ss);
        if (!ss) throw std::runtime_error("Unexpected end of input");
        char ch = ss.peek();
        if (ch == '{') return parseObject(ss);
        else if (ch == '[') return parseArray(ss);
        else if (ch == '"') return parseString(ss);
        else if (std::isdigit(ch) || ch == '-') return parseNumber(ss);
        else if (std::isalpha(ch)) return parseLiteral(ss);
        throw std::runtime_error("Invalid JSON value");
    }
    std::shared_ptr<JsonValue> parse(const std::string& text){
        std::istringstream ss(text);
        return parse(ss);
    }
}