#include "unicode_processor.hpp"
// Include the ICU headers
#include <unicode/errorcode.h> // UError code
#include <unicode/ustring.h> // from u_strFromUTF8, u_strToUTF8

std::string unicode_processor::normaliseString(const std::string& input, UNormalizationMode mode) {
    // Convert the input string to a UChar*
    UErrorCode errorCode = U_ZERO_ERROR;
    const icu::Normalizer2* normalizer = nullptr;
    if (mode == UNORM_NFC) {
        normalizer = icu::Normalizer2::getNFKInstance(UNormalizer2Mode::NORMALIZATION_NFC, errorCode);
    } else if (mode == UNORM_NFD) {
        normalizer = icu::Normalizer2::getNFDInstance(UNormalizer2Mode::NORMALIZATION_NFD, errorCode);
    } else if (mode == UNORM_NFKC) {
        normalizer = icu::Normalizer2::getNFKCInstance(UNormalizer2Mode::NORMALIZATION_NFKC, errorCode);
    } else if (mode == UNORM_NFKD) {
        normalizer = icu::Normalizer2::getNFKDInstance(UNormalizer2Mode::NORMALIZATION_NFKD, errorCode);
    } else {
        throw std::runtime_error("Unsupported normalization mode");
    }

    if (U_FAILURE(errorCode) || normalizer == nullptr) {
        throw std::runtime_error("Failed to get normalizer instance");
    }

    // convert UTF_8 to ICU Ustring (UTF_16)
    icu::UnicodeString unicodeInput = icu::UnicodeString::fromUTF8(input);
    icu::UnicodeString uNormalized;
    normalizer->normalize(unicodeInput, uNormalized, errorCode);
    if (U_FAILURE(errorCode)) {
        throw std::runtime_error("Normalization failed");
    }
    // convert back to UTF_8
    std::string normalizedString = uNormalized.toUTF8String();
    return normalizedString;
}

std::UnicodeString unicode_processor::removeDiacritics(const std::string& inputstring) {
    // Convert the input string to a UnicodeString
    icu::UnicodeString uInput = icu::UnicodeString::fromUTF8(inputstring);
    icu::UnicodeString uResult;

    // Iterate through each character in the Unicode string
    for (int32_t i = 0; i < uInput.length(); ++i) {
        UChar32 c = uInput.char32At(i);
        UCharCategory category = u_charType(c);
        // Check if the character is a combining mark
        if (category != U_COMBINING_SPACING_MARK && 
            category != U_NON_SPACING_MARK && 
            category != U_ENCLOSING_MARK) {
            // If not a combining mark, append it to the result
            uResult.append(c);
        } else {
            // If it is a combining mark, skip it
            continue;
        }
        i += icu::U16_LENGTH(C) // iterare through the string correctly
    }
    // convert ICU's UnicodeString back to a std::string
    std::string output;
    uResult.toUTF8String(output);
    return output;
}