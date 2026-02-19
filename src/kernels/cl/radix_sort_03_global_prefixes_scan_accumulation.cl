#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Down-pass: add per-chunk offsets (uint4) to scanned array.
// scan_inout4 and upper_scan4 are packed uint4 arrays in uint[].
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global       uint* scan_inout4,   // packed uint4, length = len*4
    __global const uint* upper_scan4,   // packed uint4, length = ceil(len/512)*4
    unsigned int len,
    unsigned int block_size)
{
    (void)block_size; // assumes 512, kept for host-side signature uniformity

    const uint lid = (uint)get_local_id(0);
    const uint gid = (uint)get_group_id(0);

    const uint base = gid * 512u;
    const uint idx0 = base + 2u * lid;
    const uint idx1 = idx0 + 1u;

    uint4 offset = (uint4)(0u, 0u, 0u, 0u);
    if (gid > 0u) {
        offset = vload4(gid - 1u, upper_scan4);
    }

    if (idx0 < len) {
        const uint4 v = vload4(idx0, scan_inout4);
        vstore4(v + offset, idx0, scan_inout4);
    }
    if (idx1 < len) {
        const uint4 v = vload4(idx1, scan_inout4);
        vstore4(v + offset, idx1, scan_inout4);
    }
}
