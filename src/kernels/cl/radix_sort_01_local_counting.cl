#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Radix sort pass (RADIX=4, 2 bits):
// For each 512-element block (work-group of 256), compute local histogram of digits (0..3).
// Output layout: counts4[block] is stored as 4 uints (packed uint4) in counts4_out.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,        // length = n
    __global       uint* counts4_out,   // length = num_blocks * 4 (packed uint4 per block)
    unsigned int n,
    unsigned int shift)
{
    const uint lid = (uint)get_local_id(0);
    const uint gid = (uint)get_group_id(0);

    const uint base = gid * 512u;
    const uint i0 = base + 2u * lid;
    const uint i1 = i0 + 1u;

    uint4 c = (uint4)(0u, 0u, 0u, 0u);
    if (i0 < n) {
        const uint v0 = values[i0];
        const uint d0 = (v0 >> shift) & 3u;
        if (d0 == 0u) c.s0 += 1u;
        else if (d0 == 1u) c.s1 += 1u;
        else if (d0 == 2u) c.s2 += 1u;
        else c.s3 += 1u;
    }
    if (i1 < n) {
        const uint v1 = values[i1];
        const uint d1 = (v1 >> shift) & 3u;
        if (d1 == 0u) c.s0 += 1u;
        else if (d1 == 1u) c.s1 += 1u;
        else if (d1 == 2u) c.s2 += 1u;
        else c.s3 += 1u;
    }

    __local uint4 tmp[GROUP_SIZE];
    tmp[lid] = c;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce 256 -> 1 (component-wise) to get histogram for the 512-element block.
    for (uint stride = GROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            tmp[lid] += tmp[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0u) {
        vstore4(tmp[0], gid, counts4_out);
    }
}
