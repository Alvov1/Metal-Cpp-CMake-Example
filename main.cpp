#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "MetalException.h"

int main() {
    try {
        MTL::Device* device = MTL::CreateSystemDefaultDevice();
        NS::Error* pError = nullptr;

        /* 1. Load source-code of the Metal library into string. */
        const std::filesystem::path kernelPath = "../kernel.metal";
        const auto librarySource = [&kernelPath] {
            std::ifstream source(kernelPath);
            return std::string((std::istreambuf_iterator<char>(source)),{});
        } ();

        /* 2. Create metal library object. */
        MTL::Library* library = device->newLibrary( NS::String::string(librarySource.c_str(), NS::ASCIIStringEncoding), nullptr, &pError);
        if(library == nullptr || pError != nullptr)
            throw MetalException(pError);

        /* 3. Load kernel function from the library. */
        const NS::String* fName = NS::String::string("add_arrays", NS::ASCIIStringEncoding);
        const MTL::Function* ourFunction = library->newFunction(fName, nullptr, &pError);
        if(ourFunction == nullptr || pError != nullptr)
            throw MetalException(pError);

        /* 4. Prepare data. */
        const unsigned arrayLength = 8;
        using BufferType = unsigned;
        const std::tuple buffers = {
                device->newBuffer(std::vector<BufferType>{ 0, 1, 2, 3, 4, 5, 6, 7 }.data(), arrayLength * sizeof(BufferType), MTL::ResourceStorageModeShared),
                device->newBuffer(std::vector<BufferType>{ 8, 9, 10, 11, 12, 13, 14, 15 }.data(), arrayLength * sizeof(BufferType), MTL::ResourceStorageModeShared),
                device->newBuffer(arrayLength * sizeof(BufferType), MTL::ResourceStorageModeShared)
        }; const auto& [bufferA, bufferB, resultBuffer] = buffers;

        /* 5. Init required settings. */
        const MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(ourFunction, &pError);
        if(pipelineState == nullptr || pError != nullptr)
            throw MetalException(pError);
        MTL::CommandQueue* commandQueue = device->newCommandQueue();
        if(commandQueue == nullptr)
            throw std::runtime_error("0. Failed to prepare command queue object.");
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        if(commandBuffer == nullptr)
            throw std::runtime_error("0. Failed to allocate command buffer.");

        /* 6. Prepare command encoder. */
        MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
        if(commandEncoder == nullptr)
            throw std::runtime_error("0. Failed to generate commandEncoder.");
        commandEncoder->setComputePipelineState(pipelineState);
        commandEncoder->setBuffer(bufferA, 0, 0);
        commandEncoder->setBuffer(bufferB, 0, 1);
        commandEncoder->setBuffer(resultBuffer, 0, 2);

        /* 7. Launch threads. */
        const NS::UInteger threadGroupSize =
                pipelineState->maxTotalThreadsPerThreadgroup() > arrayLength ?
                pipelineState->maxTotalThreadsPerThreadgroup() : arrayLength;
        commandEncoder->dispatchThreads(MTL::Size(arrayLength, 1, 1), MTL::Size(threadGroupSize, 1, 1));
        commandEncoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        /* 8. View results. */
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