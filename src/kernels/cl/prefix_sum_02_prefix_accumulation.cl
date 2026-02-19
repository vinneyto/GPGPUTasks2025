#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Distribute (down-sweep) block-prefix offsets:
// For each element i in a 512-block, add offset = upper_scan[blockId-1] (inclusive prefix of block sums).
// This turns per-block scan into a global scan for this level.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global uint* scan_inout,         // length = n, per-block inclusive scan; will become global inclusive scan
    __global const uint* upper_scan,   // length = ceil(n / block_size), global inclusive scan of block sums
    unsigned int n,
    unsigned int block_size)
{
    // Each work-group corresponds to one 512-element block (2 elements per thread).
    const uint lid = (uint)get_local_id(0);
    const uint gid = (uint)get_group_id(0);

    const uint base = gid * (uint)(2u * GROUP_SIZE); // == gid * 512
    const uint i0 = base + 2u * lid;
    const uint i1 = i0 + 1u;

    // The first block has zero offset.
    uint offset = 0u;
    if (gid > 0u) {
        // upper_scan is inclusive; to get sum of all previous blocks use [gid-1]
        offset = upper_scan[gid - 1u];
    }

    // Keep block_size parameter for host-side consistency; kernel logic assumes block_size == 512.
    (void)block_size;

    if (i0 < n) scan_inout[i0] += offset;
    if (i1 < n) scan_inout[i1] += offset;
}
