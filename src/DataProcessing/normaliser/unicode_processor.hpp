#ifndef UNICODE_PROCESSOR_HPP
#define UNICODE_PROCESSOR_HPP

#include <string>
#include <stdexcept>

// ICU headers
#include <unicode/unistr.h>       // UnicodeString
#include <unicode/uchar.h>        // u_charType
#include <unicode/normalizer2.h>  // Normalizer2
#include <unicode/errorcode.h>    // ErrorCode

enum class NormalizationMode {
    NFC,
    NFD,
    NFKC,
    NFKD
};

class UnicodeProcessor {
public:
    static std::string normaliseString(const std::string& input, NormalizationMode mode);
    static std::string removeDiacritics(const std::string& input);
private:
    UnicodeProcessor() = delete;
};

#endif // UNICODE_PROCESSOR_HPP