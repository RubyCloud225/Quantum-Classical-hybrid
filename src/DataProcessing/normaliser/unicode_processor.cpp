#include "unicode_processor.hpp"
#include "utils/logger.hpp"
#include <omp.h>
#include <vector>

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

    // OpenMP parallel remove diacritics
    int nThreads = omp_get_max_threads();
    std::vector<icu::UnicodeString> threadResults(nThreads);

    // Precompute the start indices of each code point
    std::vector<int32_t> indices;
    for (int32_t i = 0; i < uInput.length(); ) {
        indices.push_back(i);
        UChar32 c = uInput.char32At(i);
        i += U16_LENGTH(c);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& localResult = threadResults[tid];

        #pragma omp for
        for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
            int32_t idx = indices[i];
            UChar32 c = uInput.char32At(idx);
            UCharCategory cat = static_cast<UCharCategory>(u_charType(c));
            if (cat != U_COMBINING_SPACING_MARK && cat != U_NON_SPACING_MARK && cat != U_ENCLOSING_MARK) {
                localResult.append(c);
            }
        }
    }

    for (const auto& local : threadResults) {
        uResult.append(local);
    }
    Logger::log("Removing diacritics from string: " + input, LogLevel::INFO, __FILE__, __LINE__);

    std::string result;
    uResult.toUTF8String(result);
    return result;
}