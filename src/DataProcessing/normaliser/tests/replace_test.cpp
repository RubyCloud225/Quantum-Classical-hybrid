#include "replace.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <regex>

void printTestStatus(const std::string& testName, bool status) {
    std::cout << (status ? "PASS: " : "FAIL: ") << testName << std::endl;
}

int main() {
    
    std::cout << "--- Running Replace Class Tests ---" << std::endl;
    // basic test for Replace class
    {
        std::string testName = "Test Replace Basic Functionality";
        Replace replace("foo", "bar");
        std::string input = "foo foo foo";
        std::string expectedOutput = "bar bar bar";
        std::string actualOutput = Replace.replace(input);
        printTestStatus(testName, actualOutput == expectedOutput);
        assert(actualOutput == expectedOutput);
    }
    // edge case: no matches found
    {
        std::string testName = "Test Replace No Matches Found";
        Replace replace("xyz", "abc");
        std::string input = "foo bar baz";
        std::string expectedOutput = "foo bar baz";
        std::string actualOutput = Replace.replace(input);
        printTestStatus(testName, actualOutput == expectedOutput);
        assert(actualOutput == expectedOutput);
    }
    // edge case: multiple matches found
    {
        std::string testName = "Test Replace Multiple Matches Found";
        Replace replace("foo", "bar");
        std::string input = "foo foo foo";
        std::string expectedOutput = "bar bar bar";
        std::string actualOutput = Replace.replace(input);
        printTestStatus(testName, actualOutput == expectedOutput);
        assert(actualOutput == expectedOutput);
    }
    // edge case: regex pattern is empty
    {
        std::string testName = "Test Replace Regex Pattern Empty";
        Replace replace("", "bar");
        std::string input = "foo bar baz";
        std::string expectedOutput = "foo bar baz";
        std::string actualOutput = Replace.replace(input);
        printTestStatus(testName, actualOutput == expectedOutput);
        assert(actualOutput == expectedOutput);
    }
    // edge case: replace string is empty
    {
        std::string testName = "Test Replace Replace String Empty";
        Replace replace("foo", "");
        std::string input = "foo bar baz";
        std::string expectedOutput = " bar baz";
        std::string actualOutput = Replace.replace(input);
        printTestStatus(testName, actualOutput == expectedOutput);
        assert(actualOutput == expectedOutput);
    }
    // edge case: regex special characters
    {
        std::string testName = "Test Replace Regex Special Characters";
        Replace replace("a\\b", "c");
        std::string input = "a\\b";
        std::string expectedOutput = "c";
        std::string actualOutput = Replace.replace(input);
        printTestStatus(testName, actualOutput == expectedOutput);
        assert(actualOutput == expectedOutput);
    }
    // edge case: case sensitivity
    {
        std::string testName = "Test Replace Case Sensitivity";
        Replace replace("FOO", "bar");
        std::string input = "FOO bar baz";
        std::string expectedOutput = "FOO bar baz";
        std::string actualOutput = Replace.replace(input);
        printTestStatus(testName, actualOutput == expectedOutput);
        assert(actualOutput == expectedOutput);
    }
    // edge case: non-ASCII characters
    {
        std::string testName = "Test Replace Non-ASCII Characters";
        Replace replace("ä", "a");
        std::string input = "föö bar baz";
        std::string expectedOutput = "föö bar baz";
        std::string actualOutput = Replace.replace(input);
        printTestStatus(testName, actualOutput == expectedOutput);
        assert(actualOutput == expectedOutput);
    }
    // Clone Test Functionality 
    {
        std::string testName = "Test Clone";
        Replace replace("foo", "bar");
        Replace clone = replace.clone();
        std::string input = "foo bar baz";
        std::string expectedOutput = "bar bar baz";
        std::string actualOutput = clone.replace(input);
        printTestStatus(testName, actualOutput == expectedOutput);
        assert(actualOutput == expectedOutput);
    }
    // Serialise and Deserialise Test Functionality
    {
        std::string testName = "Test Serialise and Deserialise";
        Replace replace("foo", "bar");
        std::string serialisedData = replace.serialise();
        Replace deserialisedReplace = Replace::deserialise(serialisedData);
        std::string input = "foo bar baz";
        std::string expectedOutput = "bar bar baz";
        std::string actualOutput = deserialisedReplace.replace(input);
        printTestStatus(testName, actualOutput == expectedOutput);
        assert(actualOutput == expectedOutput);
    }
    // Deserialize Invalid Data Test Functionality
    {
        std::string testName = "Test Deserialize Invalid Data";
        std::string invalidData = "Invalid data";
        Replace deserialisedReplace = Replace::deserialise(invalidData);
        std::string input = "foo bar baz";
        std::string expectedOutput = "foo bar baz";
        std::string actualOutput = deserialisedReplace.replace(input);
        printTestStatus(testName, actualOutput == expectedOutput);
        assert(actualOutput == expectedOutput);
    }
    // Deserialize Empty Data Test Functionality
    {
        std::string testName = "Test Deserialize Empty Data";
        std::string emptyData = "";
        Replace deserialisedReplace = Replace::deserialise(emptyData);
        std::string input = "foo bar baz";
        std::string expectedOutput = "foo bar baz";
        std::string actualOutput = deserialisedReplace.replace(input);
        printTestStatus(testName, actualOutput == expectedOutput);
        assert(actualOutput == expectedOutput);
    }
    std::cout << "--- All Tests Completed ---" << std::endl;
}
