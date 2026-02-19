#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Scatter for RADIX=4 (2 bits per pass).
// values_out is filled using global block-prefixes (scan_counts4) + local stable ranks.
// scan_counts4 is packed uint4 inclusive scan of per-block digit counts (length = num_blocks*4).
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values_in,      // length = n
    __global const uint* scan_counts4,   // length = num_blocks*4 (packed uint4 per block)
    __global       uint* values_out,     // length = n
    unsigned int n,
    unsigned int shift)
{
    const uint lid = (uint)get_local_id(0);
    const uint gid = (uint)get_group_id(0); // block id

    const uint base = gid * 512u;
    const uint i0 = base + 2u * lid;
    const uint i1 = i0 + 1u;

    // Cache global_starts for this block in local memory (avoids redundant vload4 for all threads).
    __local uint4 global_starts_local;
    if (lid == 0u) {
        const uint num_blocks = (n + 511u) / 512u;
        uint4 totals = (uint4)(0u, 0u, 0u, 0u);
        if (num_blocks > 0u) {
            totals = vload4(num_blocks - 1u, scan_counts4);
        }
        const uint base0 = 0u;
        const uint base1 = totals.s0;
        const uint base2 = totals.s0 + totals.s1;
        const uint base3 = totals.s0 + totals.s1 + totals.s2;
        const uint4 digit_bases = (uint4)(base0, base1, base2, base3);

        uint4 prefix_before = (uint4)(0u, 0u, 0u, 0u);
        if (gid > 0u) {
            prefix_before = vload4(gid - 1u, scan_counts4);
        }
        global_starts_local = digit_bases + prefix_before;
    }

    // digits for the two elements
    uint v0 = 0u, v1 = 0u;
    uint d0 = 0u, d1 = 0u;
    uint valid0 = 0u, valid1 = 0u;
    if (i0 < (uint)n) {
        valid0 = 1u;
        v0 = values_in[i0];
        d0 = (v0 >> shift) & 3u;
    }
    if (i1 < (uint)n) {
        valid1 = 1u;
        v1 = values_in[i1];
        d1 = (v1 >> shift) & 3u;
    }

    // Per-thread digit counts vector (for stable local ranks)
    uint4 c = (uint4)(0u, 0u, 0u, 0u);
    if (valid0) {
        if (d0 == 0u) c.s0 += 1u;
        else if (d0 == 1u) c.s1 += 1u;
        else if (d0 == 2u) c.s2 += 1u;
        else c.s3 += 1u;
    }
    if (valid1) {
        if (d1 == 0u) c.s0 += 1u;
        else if (d1 == 1u) c.s1 += 1u;
        else if (d1 == 2u) c.s2 += 1u;
        else c.s3 += 1u;
    }

    // Exclusive scan of c across 256 threads (component-wise via uint4)
    __local uint4 temp[GROUP_SIZE];
    temp[lid] = c;
    barrier(CLK_LOCAL_MEM_FENCE); // also sync global_starts_local

    // up-sweep
    for (uint offset = 1u; offset < GROUP_SIZE; offset <<= 1u) {
        const uint idx = (lid + 1u) * (offset << 1u) - 1u;
        if (idx < GROUP_SIZE) {
            temp[idx] += temp[idx - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // exclusive
    if (lid == 0u) {
        temp[GROUP_SIZE - 1u] = (uint4)(0u, 0u, 0u, 0u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // down-sweep
    for (uint offset = GROUP_SIZE / 2u; offset > 0u; offset >>= 1u) {
        const uint idx = (lid + 1u) * (offset << 1u) - 1u;
        if (idx < GROUP_SIZE) {
            const uint4 t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint4 thread_prefix = temp[lid];
    const uint4 global_starts = global_starts_local;

    if (valid0) {
        uint local_rank0 = 0u;
        uint dst0 = 0u;
        if (d0 == 0u) {
            local_rank0 = thread_prefix.s0;
            dst0 = global_starts.s0 + local_rank0;
        } else if (d0 == 1u) {
            local_rank0 = thread_prefix.s1;
            dst0 = global_starts.s1 + local_rank0;
        } else if (d0 == 2u) {
            local_rank0 = thread_prefix.s2;
            dst0 = global_starts.s2 + local_rank0;
        } else {
            local_rank0 = thread_prefix.s3;
            dst0 = global_starts.s3 + local_rank0;
        }
        values_out[dst0] = v0;
    }

    if (valid1) {
        const uint add = (valid0 && (d1 == d0)) ? 1u : 0u;
        uint local_rank1 = 0u;
        uint dst1 = 0u;
        if (d1 == 0u) {
            local_rank1 = thread_prefix.s0 + add;
            dst1 = global_starts.s0 + local_rank1;
        } else if (d1 == 1u) {
            local_rank1 = thread_prefix.s1 + add;
            dst1 = global_starts.s1 + local_rank1;
        } else if (d1 == 2u) {
            local_rank1 = thread_prefix.s2 + add;
            dst1 = global_starts.s2 + local_rank1;
        } else {
            local_rank1 = thread_prefix.s3 + add;
            dst1 = global_starts.s3 + local_rank1;
        }
        values_out[dst1] = v1;
    }
}