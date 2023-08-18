#include <iostream>
#include <sstream>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <iomanip>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

class MetalException final: public std::exception {
    std::string message {};
public:
    explicit MetalException(const NS::Error* ofError) {
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
    [[nodiscard]] const char* what() const noexcept override {
        return message.c_str();
    }
};

template <typename bufferType = float>
MTL::Buffer* makeRandomBuffer(MTL::Device* onDevice, unsigned withSize) {
    MTL::Buffer* ourBuffer = onDevice->newBuffer(withSize * sizeof(bufferType), MTL::ResourceStorageModeShared);
    if(ourBuffer == nullptr)
        throw std::runtime_error("0. Failed to allocate buffer on device.");
    for(unsigned i = 0; i < withSize; ++i)
        reinterpret_cast<bufferType*>(ourBuffer->contents())[i] = rand();
    return ourBuffer;
}

int main() {
    try {
        MTL::Device* device = MTL::CreateSystemDefaultDevice();
        NS::Error* pError = nullptr;

        const char* kernelSrc = R"(
            #include <metal_stdlib>
            using namespace metal;

            kernel void add_arrays(device const unsigned* inA, device const unsigned* inB, device unsigned* result, uint index [[thread_position_in_grid]]){
                result[index] = inA[index] + inB[index];
            }
        )";

        /* 1. Load library with kernel function. */
        MTL::Library* library = device->newLibrary( NS::String::string(kernelSrc, NS::UTF8StringEncoding), nullptr, &pError);
        if(library == nullptr || pError != nullptr)
            throw MetalException(pError);

        /* 2. Load kernel function from the library. */
        const NS::String* fName = NS::String::string("add_arrays", NS::ASCIIStringEncoding);
        const MTL::Function* ourFunction = library->newFunction(fName, nullptr, &pError);
        if(ourFunction == nullptr || pError != nullptr)
            throw MetalException(pError);

        /* 3. Prepare data. */
        const unsigned arrayLength = 16;
        using BufferType = unsigned;
        const std::tuple buffers = {
                makeRandomBuffer<BufferType>(device, arrayLength),
                makeRandomBuffer<BufferType>(device, arrayLength),
                device->newBuffer(arrayLength * sizeof(BufferType), MTL::ResourceStorageModeShared)
        }; const auto& [bufferA, bufferB, resultBuffer] = buffers;

        /* 4. Init required settings. */
        const MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(ourFunction, &pError);
        if(pipelineState == nullptr || pError != nullptr)
            throw MetalException(pError);
        MTL::CommandQueue* commandQueue = device->newCommandQueue();
        if(commandQueue == nullptr)
            throw std::runtime_error("0. Failed to prepare command queue object.");
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        if(commandBuffer == nullptr)
            throw std::runtime_error("0. Failed to allocate command buffer.");

        /* 5. Prepare command encoder. */
        MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
        if(commandEncoder == nullptr)
            throw std::runtime_error("0. Failed to generate commandEncoder.");
        commandEncoder->setComputePipelineState(pipelineState);
        commandEncoder->setBuffer(bufferA, 0, 0);
        commandEncoder->setBuffer(bufferB, 0, 1);
        commandEncoder->setBuffer(resultBuffer, 0, 2);

        /* 6. Launch threads. */
        const NS::UInteger threadGroupSize =
                pipelineState->maxTotalThreadsPerThreadgroup() > arrayLength ?
                pipelineState->maxTotalThreadsPerThreadgroup() : arrayLength;
        commandEncoder->dispatchThreads(MTL::Size(arrayLength, 1, 1), MTL::Size(threadGroupSize, 1, 1));
        commandEncoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        /* 7. View results. */
        std::cout << std::setw(12) << "Array A: ";
        for(unsigned i = 0; i < arrayLength; ++i)
            std::cout << std::setw(10) << reinterpret_cast<const BufferType*>(bufferA->contents())[i] << ' ';
        std::cout << std::endl << std::setw(12) << "Array B: ";
        for(unsigned i = 0; i < arrayLength; ++i)
            std::cout << std::setw(10) << reinterpret_cast<const BufferType*>(bufferB->contents())[i] << ' ';
        std::cout << std::endl << std::setw(12) << "In common: ";
        for(unsigned i = 0; i < arrayLength; ++i)
            std::cout << std::setw(10) << reinterpret_cast<const BufferType*>(resultBuffer->contents())[i] << ' ';
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
    }
    return 0;
}