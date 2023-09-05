#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device const unsigned* inA, device const unsigned* inB, device unsigned* result, uint index [[thread_position_in_grid]]){
    result[index] = inA[index] + inB[index];
}