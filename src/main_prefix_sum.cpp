#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

static std::vector<unsigned int> calc_up_pass_level_sizes(unsigned int n, unsigned int block_size)
{
    rassert(block_size > 0, 5462345234990, block_size);

    std::vector<unsigned int> level_sizes;
    level_sizes.push_back(n); // level 0 is the original array length

    while (level_sizes.back() > block_size) {
        const unsigned int prev = level_sizes.back();
        const unsigned int next = (prev + block_size - 1) / block_size; // ceil(prev / block_size)
        level_sizes.push_back(next);
    }
    return level_sizes;
}

struct ScanLevel {
    unsigned int n = 0; // number of uint elements in this level
    gpu::gpu_mem_32u sum; // input of this level (and output_block_sums from previous)
    gpu::gpu_mem_32u scan; // output_scan of this level
};

static void verify_level(
    const std::vector<ScanLevel>& levels,
    size_t level_id,
    unsigned int block_size,
    const gpu::gpu_mem_32u& dummy_top_block_sums_gpu)
{
    rassert(block_size > 0, 5462345234994, block_size);
    rassert(level_id < levels.size(), 5462345234995, level_id, levels.size());

    const std::vector<unsigned int> cpu_input = levels[level_id].sum.readVector(levels[level_id].n);
    const std::vector<unsigned int> gpu_scan = levels[level_id].scan.readVector(levels[level_id].n);
    rassert(cpu_input.size() == gpu_scan.size(), 5462345234996, level_id, cpu_input.size(), gpu_scan.size());

    const bool has_next = (level_id + 1 < levels.size());
    std::vector<unsigned int> gpu_block_sums;
    if (has_next) {
        gpu_block_sums = levels[level_id + 1].sum.readVector(levels[level_id + 1].n);
        const size_t expected_blocks = (cpu_input.size() + block_size - 1) / block_size;
        rassert(gpu_block_sums.size() == expected_blocks, 5462345234997, level_id, gpu_block_sums.size(), expected_blocks);
    } else {
        // only to make it explicit that we intentionally ignore the last kernel's block_sums output
        rassert(dummy_top_block_sums_gpu.number() == 1, 5462345234998, dummy_top_block_sums_gpu.number());
    }

    size_t cpu_sum = 0;
    size_t block_sum = 0;
    size_t current_block = 0;
    for (size_t i = 0; i < cpu_input.size(); ++i) {
        if (i % block_size == 0) {
            cpu_sum = 0;
            block_sum = 0;
            current_block = i / block_size;
        }

        cpu_sum += cpu_input[i];
        block_sum += cpu_input[i];
        rassert(cpu_sum < std::numeric_limits<unsigned int>::max(), 5462345234999, level_id, cpu_sum, i);
        rassert(block_sum < std::numeric_limits<unsigned int>::max(), 5462345235000, level_id, block_sum, i);

        rassert((unsigned int)cpu_sum == gpu_scan[i], 5462345235001, level_id, (unsigned int)cpu_sum, gpu_scan[i], i);

        const bool is_block_end = ((i % block_size) == (block_size - 1)) || ((i + 1) == cpu_input.size());
        if (is_block_end && has_next) {
            rassert(current_block < gpu_block_sums.size(), 5462345235002, level_id, current_block, gpu_block_sums.size());
            rassert((unsigned int)block_sum == gpu_block_sums[current_block],
                5462345235003, level_id, (unsigned int)block_sum, gpu_block_sums[current_block], i);
        }
    }
}

static std::vector<ScanLevel> allocate_scan_levels(unsigned int n_input, unsigned int block_size)
{
    std::vector<unsigned int> sizes = calc_up_pass_level_sizes(n_input, block_size);

    std::vector<ScanLevel> levels;
    levels.reserve(sizes.size());
    for (unsigned int s : sizes) {
        ScanLevel lvl;
        lvl.n = s;
        lvl.sum = gpu::gpu_mem_32u(s);
        lvl.scan = gpu::gpu_mem_32u(s);
        levels.push_back(std::move(lvl));
    }

    return levels;
}

void run(int argc, char** argv)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DCUDA_SUPPORT=ON в CMake options
    // TODO 010 P.S. так же в случае CUDA - добавьте в CMake options (НЕ меняйте сами CMakeLists.txt чтобы не менять окружение тестирования):
    // TODO 010 "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo" (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_fill_with_zeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_sum_reduction(ocl::getPrefixSum01Reduction());
    ocl::KernelSource ocl_prefix_accumulation(ocl::getPrefixSum02PrefixAccumulation());

    avk2::KernelSource vk_fill_with_zeros(avk2::getFillBufferWithZeros());
    avk2::KernelSource vk_sum_reduction(avk2::getPrefixSum01Reduction());
    avk2::KernelSource vk_prefix_accumulation(avk2::getPrefixSum02PrefixAccumulation());

    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    size_t total_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        as[i] = (3 * (i + 5) + 7) % 17;
        total_sum += as[i];
        rassert(total_sum < std::numeric_limits<unsigned int>::max(), 5462345234231, total_sum, as[i], i); // ensure no overflow
    }

    // Аллоцируем буферы в VRAM
    const unsigned int block_size = 512;
    const unsigned int num_blocks = (n + block_size - 1) / block_size;

    std::vector<ScanLevel> levels = allocate_scan_levels(n, block_size);
    rassert(levels.size() >= 2, 5462345234991, n, block_size, levels.size());
    rassert(levels[0].n == n, 5462345234992, levels[0].n, n);
    rassert(levels[1].n == num_blocks, 5462345234993, levels[1].n, num_blocks);

    std::cout << "buffers layout:" << std::endl;
    for (size_t i = 0; i < levels.size(); ++i) {
        std::cout << "  level " << i
                  << ": sum(n=" << levels[i].n << "), scan(n=" << levels[i].n << ")"
                  << std::endl;
    }

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    levels[0].sum.writeN(as.data(), n);

    // Заглушка для output_block_sums на самом верхнем уровне
    gpu::gpu_mem_32u dummy_top_block_sums_gpu(1);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
        // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
        if (context.type() == gpu::Context::TypeOpenCL) {
            // Up-pass: строим sums для верхних уровней и per-block scan на каждом уровне
            for (size_t level = 0; level < levels.size(); ++level) {
                const unsigned int in_n = levels[level].n;
                const unsigned int out_blocks = (in_n + block_size - 1) / block_size; // ceil(in_n / 512)

                // Один work-group (256 потоков) обрабатывает 512 элементов.
                // global_work_size = out_blocks * 256
                gpu::WorkSize workSize(GROUP_SIZE, (size_t)out_blocks * GROUP_SIZE);

                gpu::gpu_mem_32u& out_block_sums = (level + 1 < levels.size()) ? levels[level + 1].sum : dummy_top_block_sums_gpu;

                ocl_sum_reduction.exec(workSize, levels[level].sum, levels[level].scan, out_block_sums, in_n);
            }

            // Down-pass: распространяем оффсеты сверху вниз (делаем scan глобальным на каждом уровне)
            for (size_t level = levels.size() - 1; level-- > 0;) {
                // level goes: last-1, ..., 0
                const unsigned int in_n = levels[level].n;
                const unsigned int out_blocks = (in_n + block_size - 1) / block_size; // ceil(in_n / 512)
                gpu::WorkSize ws(GROUP_SIZE, (size_t)out_blocks * GROUP_SIZE);
                ocl_prefix_accumulation.exec(ws, levels[level].scan, levels[level + 1].scan, levels[level].n, block_size);
            }
        } else if (context.type() == gpu::Context::TypeCUDA) {
            // TODO
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // cuda::fill_buffer_with_zeros();
            // cuda::prefix_sum_01_sum_reduction();
            // cuda::prefix_sum_02_prefix_accumulation();
        } else if (context.type() == gpu::Context::TypeVulkan) {
            // TODO
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // vk_fill_with_zeros.exec();
            // vk_sum_reduction.exec();
            // vk_prefix_accumulation.exec();
        } else {
            rassert(false, 4531412341, context.type());
        }

        times.push_back(t.elapsed());
    }
    std::cout << "prefix sum times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // "Наивная" метрика пропускной способности как в самом начале:
    // считаем, что задача = прочитать n uint и записать n uint (2*N*4 байта).
    const double memory_size_gb = sizeof(unsigned int) * 2.0 * n / 1024.0 / 1024.0 / 1024.0;
    const double median_time = stats::median(times);
    std::cout << "prefix sum median effective VRAM bandwidth: " << memory_size_gb / median_time << " GB/s" << std::endl;

    // Пропускная способность по входным элементам (сколько uint/сек обрабатываем)
    const double elems_per_sec = (double)n / median_time;
    std::cout << "prefix sum throughput: " << (elems_per_sec / 1e9) << " Guint/s" << std::endl;

    //    // Проверяем любой уровень (можно поставить конкретный level_id для отладки),
    //    // а сейчас пробегаем все уровни.
    //    for (size_t level = 0; level < levels.size(); ++level) {
    //        verify_level(levels, level, block_size, dummy_top_block_sums_gpu);
    //    }

    // Итоговая проверка: полный inclusive prefix sum по всему массиву
    std::vector<unsigned int> gpu_prefix_sum = levels[0].scan.readVector();
    size_t cpu_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        cpu_sum += as[i];
        rassert(cpu_sum < std::numeric_limits<unsigned int>::max(), 5462345236000, cpu_sum, i);
        rassert((unsigned int)cpu_sum == gpu_prefix_sum[i], 5462345236001, (unsigned int)cpu_sum, gpu_prefix_sum[i], i);
    }

    // Проверяем что входные данные остались нетронуты (ведь мы их переиспользуем от итерации к итерации)
    std::vector<unsigned int> input_values = levels[0].sum.readVector();
    for (size_t i = 0; i < n; ++i) {
        rassert(input_values[i] == as[i], 6573452432, input_values[i], as[i]);
    }
}

int main(int argc, char** argv)
{
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за выбора CUDA API (его нет на процессоре - т.е. в случае CI на GitHub Actions)
            return 0;
        }
        if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            return 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            return 1;
        }
    }

    return 0;
}
