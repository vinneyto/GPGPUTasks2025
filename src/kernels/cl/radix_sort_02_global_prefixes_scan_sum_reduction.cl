#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Block scan (Blelloch) for array of uint4 packed into uint[4*len].
// One work-group (256 threads) scans 512 elements (2 per thread).
// Writes inclusive scan to output_scan4 and per-chunk sums to output_block_sums4.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* input4,            // packed uint4 array, length = len*4
    __global       uint* output_scan4,      // packed uint4 array, length = len*4
    __global       uint* output_block_sums4,// packed uint4 array, length = ceil(len/512)*4
    unsigned int len)
{
    const uint lid = (uint)get_local_id(0);
    const uint gid = (uint)get_group_id(0);

    const uint base = gid * 512u;
    const uint idx0 = base + 2u * lid;
    const uint idx1 = idx0 + 1u;

    uint4 v0 = (uint4)(0u, 0u, 0u, 0u);
    uint4 v1 = (uint4)(0u, 0u, 0u, 0u);
    if (idx0 < len) v0 = vload4(idx0, input4);
    if (idx1 < len) v1 = vload4(idx1, input4);

    __local uint4 temp[512];
    temp[2u * lid]     = v0;
    temp[2u * lid + 1] = v1;
    barrier(CLK_LOCAL_MEM_FENCE);

    // up-sweep
    for (uint offset = 1u; offset < 512u; offset <<= 1u) {
        const uint idx = (lid + 1u) * (offset << 1u) - 1u;
        if (idx < 512u) {
            temp[idx] += temp[idx - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // exclusive scan
    if (lid == 0u) {
        temp[511] = (uint4)(0u, 0u, 0u, 0u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // down-sweep
    for (uint offset = 256u; offset > 0u; offset >>= 1u) {
        const uint idx = (lid + 1u) * (offset << 1u) - 1u;
        if (idx < 512u) {
            const uint4 t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write inclusive scan
    if (idx0 < len) vstore4(temp[2u * lid] + v0, idx0, output_scan4);
    if (idx1 < len) vstore4(temp[2u * lid + 1u] + v1, idx1, output_scan4);

    // chunk sum (inclusive last element)
    if (lid == (GROUP_SIZE - 1u)) {
        vstore4(temp[511] + v1, gid, output_block_sums4);
    }
}
