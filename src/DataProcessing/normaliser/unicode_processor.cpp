#include "unicode_processor.hpp"
#include "utils/logger.hpp"

std::string UnicodeProcessor::normaliseString(const std::string& input, NormalizationMode mode) {
    UErrorCode errorCode = U_ZERO_ERROR;
    const icu::Normalizer2* normalizer = nullptr;

    switch (mode) {
        case NormalizationMode::NFC:
            normalizer = icu::Normalizer2::getNFCInstance(errorCode);
            break;
        case NormalizationMode::NFD:
            normalizer = icu::Normalizer2::getNFDInstance(errorCode);
            break;
        case NormalizationMode::NFKC:
            normalizer = icu::Normalizer2::getNFKCInstance(errorCode);
            break;
        case NormalizationMode::NFKD:
            normalizer = icu::Normalizer2::getNFKDInstance(errorCode);
            break;
        default:
            throw std::runtime_error("Unsupported normalization mode");
    }
    Logger::log("Using normalization mode: " + std::to_string(static_cast<int>(mode)), LogLevel::INFO, __FILE__, __LINE__);

    if (U_FAILURE(errorCode) || !normalizer) {
        throw std::runtime_error("Failed to get normalizer instance");
    }
    Logger::log("Normalizing string: " + input, LogLevel::INFO, __FILE__, __LINE__);

    icu::UnicodeString unicodeInput = icu::UnicodeString::fromUTF8(input);
    icu::UnicodeString normalized;
    normalizer->normalize(unicodeInput, normalized, errorCode);

    if (U_FAILURE(errorCode)) {
        throw std::runtime_error("Normalization failed");
    }
    Logger::log("Normalization completed successfully", LogLevel::INFO, __FILE__, __LINE__);

    std::string result;
    normalized.toUTF8String(result);
    return result;
}

std::string UnicodeProcessor::removeDiacritics(const std::string& input) {
    icu::UnicodeString uInput = icu::UnicodeString::fromUTF8(input);
    icu::UnicodeString uResult;

    for (int32_t i = 0; i < uInput.length(); ) {
        UChar32 c = uInput.char32At(i);
        UCharCategory cat = static_cast<UCharCategory>(u_charType(c));

        if (cat != U_COMBINING_SPACING_MARK && cat != U_NON_SPACING_MARK && cat != U_ENCLOSING_MARK) {
            uResult.append(c);
        }
        i += U16_LENGTH(c);
    }
    Logger::log("Removing diacritics from string: " + input, LogLevel::INFO, __FILE__, __LINE__);

    std::string result;
    uResult.toUTF8String(result);
    return result;
}