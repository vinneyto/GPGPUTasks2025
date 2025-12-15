#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global       uint* sum,
                                                       const unsigned int n)
{
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    __local uint local_data[GROUP_SIZE];

    if (index >= n)
        return;

    // Параллельная загрузка из глобальной памяти в локальную (все 256 потоков)
    local_data[local_index] = a[index];

    // Ждем пока все потоки в workgroup загрузят свои данные
    barrier(CLK_LOCAL_MEM_FENCE);

    // Только нулевой поток суммирует все значения из локальной памяти
    if (local_index == 0) {
        uint local_sum = 0;
        for (uint i = 0; i < GROUP_SIZE; i++) {
            local_sum += local_data[i];
        }
        // Один глобальный атомик на всю workgroup
        atomic_add(sum, local_sum);
    }
}
