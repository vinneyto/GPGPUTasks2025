#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Block scan (Blelloch) for 512 элементов на work-group из 256 потоков:
// - каждый поток грузит 2 элемента
// - на выходе: префиксы внутри каждого 512-блока + сумма блока (1 число на блок)
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,
    __global       uint* output_scan,
    __global       uint* output_block_sums,
    unsigned int n)
{
    const uint lid = (uint)get_local_id(0);
    const uint gid = (uint)get_group_id(0);

    const uint base = gid * 512u;
    const uint i0 = base + 2u * lid;
    const uint i1 = i0 + 1u;

    uint v0 = 0;
    uint v1 = 0;
    if (i0 < n) v0 = input[i0];
    if (i1 < n) v1 = input[i1];

    __local uint temp[512];
    temp[2u * lid]     = v0;
    temp[2u * lid + 1] = v1;

    barrier(CLK_LOCAL_MEM_FENCE);

    // up-sweep (reduce) phase
    for (uint offset = 1u; offset < 512u; offset <<= 1u) {
        const uint idx = (lid + 1u) * (offset << 1u) - 1u;
        if (idx < 512u) {
            temp[idx] += temp[idx - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // exclusive scan: clear the last element
    if (lid == 0u) {
        temp[511] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // down-sweep phase
    for (uint offset = 256u; offset > 0u; offset >>= 1u) {
        const uint idx = (lid + 1u) * (offset << 1u) - 1u;
        if (idx < 512u) {
            const uint t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // convert to inclusive scan and write out
    if (i0 < n) output_scan[i0] = temp[2u * lid] + v0;
    if (i1 < n) output_scan[i1] = temp[2u * lid + 1u] + v1;

    // block sum = inclusive value of the last element (valid elements only; out-of-range inputs were zeros)
    if (lid == (GROUP_SIZE - 1u)) {
        // lid==255 corresponds to index 511 (second element loaded as v1)
        output_block_sums[gid] = temp[511] + v1;
    }
}
