#ifndef METALCPPEXAMPLE_METALEXCEPTION_H
#define METALCPPEXAMPLE_METALEXCEPTION_H

#include <exception>
#include <sstream>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

class MetalException final: public std::exception {
    std::string message {};
public:
    explicit MetalException(const NS::Error* ofError);
    [[nodiscard]] const char* what() const noexcept override;
};

MetalException::MetalException(const NS::Error *ofError) {
    std::stringstream ss {};
    ss << ofError->code() << ". ";

    const auto* description = ofError->description();
    if(description != nullptr)
        ss << description->cString(NS::ASCIIStringEncoding) << ". ";

    const auto* recoverySuggestion = ofError->localizedRecoverySuggestion();
    if(recoverySuggestion != nullptr)
        ss << recoverySuggestion->cString(NS::ASCIIStringEncoding) << ". ";

    const auto* failureReason = ofError->localizedFailureReason();
    if(failureReason != nullptr)
        ss << failureReason->cString(NS::ASCIIStringEncoding) << ". ";

    message = ss.str();
}

const char *MetalException::what() const noexcept {
    return message.c_str();
}

#endif //METALCPPEXAMPLE_METALEXCEPTION_H
