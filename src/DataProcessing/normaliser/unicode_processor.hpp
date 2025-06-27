#ifndef UNICODE_PROCESSOR_HPP
#define UNICODE_PROCESSOR_HPP

#include <string> // for std::string
#include <stdexcept> // For std::runtime_error

// ICU library
#include <unicode/unistr.h> // For UnicodeString
#include <unicode/normalizer.h> // For Normalizer2
#include <unicode/uchar.h> // For Unicode character properties
#include <unicode/ustream.h> // For UnicodeString output
#include <unicode/unorm2.h> // For normalization functions

class UnicodeProcessor {
    public:
    // normalise a Unicode string to NFC form
    // returns normalised string
    static std::string normaliseString(const std::string& input, UNormalizationMode mode);
    // Remove combining diacritical marks from a Unicode string
    static std::string removeDiacritics(const std::string& input);
    private:
    UnicodeProcessor() = delete;
};

#endif // UNICODE_PROCESSOR_HPP