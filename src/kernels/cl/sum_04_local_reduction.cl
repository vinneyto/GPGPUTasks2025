#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define WARP_SIZE 32

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(__global const uint* a,
                                     __global       uint* b,
                                            unsigned int  n)
{
    const uint global_id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);
    
    __local uint local_data[GROUP_SIZE];

    // Фаза 1: Загрузка в локальную память с проверкой границ
    if (global_id < n) {
        local_data[local_id] = a[global_id];
    } else {
        local_data[local_id] = 0;  // Нейтральный элемент для потоков за границей
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Фаза 2: Reduction tree - log2(GROUP_SIZE) итераций
    for (uint stride = 1; stride < GROUP_SIZE; stride *= 2) {
        if (local_id % (2 * stride) == 0) {
            local_data[local_id] += local_data[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Фаза 3: Запись результата (один элемент на воркгруппу)
    if (local_id == 0) {
        b[group_id] = local_data[0];
    }
}
