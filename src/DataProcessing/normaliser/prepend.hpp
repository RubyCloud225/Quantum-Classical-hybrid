#ifndef PREPEND_HPP
#define PREPEND_HPP
#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <set>

class Prepend {
    public:
    Prepend(std::string filename, std::string text);
    std::set<std::string> extract_normalised(const std::string& text);
    std::string build_comment_block(const std::set<std::string>& values);
    void write_comment_block(const std::string& text, const std::string& filename);
    private:
    std::string filename;
    std::string text;
};
#endif // PREPEND_HPP