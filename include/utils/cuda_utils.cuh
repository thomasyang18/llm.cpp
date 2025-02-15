#pragma once
#include <cuda_runtime.h>
#include <iostream>

namespace user_cuda {

// Utility to check for CUDA errors after function calls
#define CHECK_CUDA_ERROR(call) {                                  \
    cudaError_t err = (call);                                     \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)     \
                  << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
}

// Memory allocation on the device using reference to pointer
template <typename T>
inline void cudaMallocDevice(T*& devPtr, size_t count) {
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&devPtr), count * sizeof(T)));
}

// Free memory on the device
template <typename T>
inline void cudaFreeDevice(T* devPtr) {
    CHECK_CUDA_ERROR(cudaFree(devPtr));
}

// Memory copy from host to device
template <typename T>
inline void cudaMemcpyHostToDevice(T* devPtr, const T* hostPtr, size_t count) {
    CHECK_CUDA_ERROR(cudaMemcpy(devPtr, hostPtr, count * sizeof(T), cudaMemcpyHostToDevice));
}

// Memory copy from device to host
template <typename T>
inline void cudaMemcpyDeviceToHost(T* hostPtr, const T* devPtr, size_t count) {
    CHECK_CUDA_ERROR(cudaMemcpy(hostPtr, devPtr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

// Memory copy from device to device
template <typename T>
inline void cudaMemcpyDeviceToDevice(T* devDest, const T* devSrc, size_t count) {
    CHECK_CUDA_ERROR(cudaMemcpy(devDest, devSrc, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

// Set memory on the device (e.g., to zero or another value)
template <typename T>
inline void cudaMemsetDevice(T* devPtr, int value, size_t count) {
    CHECK_CUDA_ERROR(cudaMemset(devPtr, value, count * sizeof(T)));
}

// Allocate unified memory accessible from both host and device
template <typename T>
inline void cudaMallocManagedDevice(T*& devPtr, size_t count) {
    CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void**>(&devPtr), count * sizeof(T)));
}

// Helper to print memory contents (for debugging)
template <typename T>
inline void printDeviceMemory(const T* devPtr, size_t count) {
    T* hostPtr = new T[count];
    cudaMemcpyDeviceToHost(hostPtr, devPtr, count);
    std::cout << "Memory contents: ";
    for (size_t i = 0; i < count; ++i) {
        std::cout << hostPtr[i] << " ";
    }
    std::cout << std::endl;
    delete[] hostPtr;
}

// Memory copy to symbol on the device
template <typename T>
inline void cudaMemcpyToSymbol(const T* symbol, const T* hostPtr, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(symbol, hostPtr, count * sizeof(T), offset, kind));
}

}
