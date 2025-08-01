// show serialize and deserialize functions for the Replace class
// provide the value for regex and replace
// take the pattern and replace every occurance with content
// clone the data
// partial template specialization for regex and replace
// Serialise the data
// Deserialize the data
// return the data

#ifndef REPLACE_HPP
#define REPLACE_HPP
#include <stdexcept>
#include <string>
#include <regex>

class Replace {
    public:
    // Constructor
    Replace(const std::string& regexPattern, const std::string& replaceWith);
    std::string applyReplace(const std::string& content) const;
    Replace clone() const;
    // std::string particularise(const std::string& pattern, const std::string& content) const;
    // add in for different types of DATA: byte, unicode, etc
    // Serialise and deserialize methods
    std::string serialise() const;
    static Replace deserialise(const std::string& serialisedData);
    // std::string replace(const std::string& content) const;
    // internal values
    std::string getPattern() const { return regexPattern; }
    std::string getReplace() const { return replaceWith; }
    private:
    std::string regexPattern; // The regex pattern to match
    std::string replaceWith; // The string to replace with
};

#endif // REPLACE_HPP