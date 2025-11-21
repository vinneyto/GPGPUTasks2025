#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

template<typename T>
std::string to_string(T value)
{
	std::ostringstream ss;
	ss << value;
	return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
	if(CL_SUCCESS == err)
		return;

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:103
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

int main()
{
	// Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
	if(!ocl_init())
		throw std::runtime_error("Can't init OpenCL driver!");

	// Откройте
	// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
	// Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
	// Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

	// Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
		// TODO 1.1
		// Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
		// Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
		// Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
		// Откройте таблицу с кодами ошибок:
		// libs/clew/CL/cl.h:103
		// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
		// Найдите там нужный код ошибки и ее название
		// Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
		// в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
		// Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
		std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of devices: " << devicesCount << std::endl;

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			cl_device_id device = devices[deviceIndex];
		    // TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));

			std::vector<unsigned char> deviceName(platformNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
			std::cout << "        Device name: " << deviceName.data() << std::endl;

			// - Тип устройства (видеокарта/процессор/что-то странное)
			cl_device_type deviceType;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
			std::cout << "        Device type: ";

			// Тип устройства - это битовая маска, поэтому нужно проверять отдельные биты
			if (deviceType & CL_DEVICE_TYPE_CPU)
				std::cout << "CPU ";
			if (deviceType & CL_DEVICE_TYPE_GPU)
				std::cout << "GPU ";
			if (deviceType & CL_DEVICE_TYPE_ACCELERATOR)
				std::cout << "ACCELERATOR ";
			if (deviceType & CL_DEVICE_TYPE_DEFAULT)
				std::cout << "(Default) ";
			if (deviceType & CL_DEVICE_TYPE_CUSTOM)
				std::cout << "CUSTOM ";

			std::cout << std::endl;

			// - Размер памяти устройства в мегабайтах
            cl_ulong globalMemSize;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMemSize, nullptr));

            // Конвертируем из байт в мегабайты
            double memSizeMB = globalMemSize / (1024.0 * 1024.0);
            std::cout << "        Global memory size: " << memSizeMB << " MB" << std::endl;


			// - Максимальная тактовая частота в MHz
			cl_uint maxClockFrequency;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &maxClockFrequency, nullptr));
			std::cout << "        Max clock frequency: " << maxClockFrequency << " MHz" << std::endl;

			// - Количество вычислительных юнитов (ядер)
			cl_uint maxComputeUnits;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, nullptr));
			std::cout << "        Max compute units: " << maxComputeUnits << std::endl;

			// - Максимальный размер рабочей группы
			size_t maxWorkGroupSize;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, nullptr));
			std::cout << "        Max work group size: " << maxWorkGroupSize << std::endl;

			// - Максимальное количество измерений для рабочих элементов
			cl_uint maxWorkItemDimensions;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWorkItemDimensions, nullptr));
			std::cout << "        Max work item dimensions: " << maxWorkItemDimensions << std::endl;

			// - Максимальный размер буфера для выделения памяти
			cl_ulong maxMemAllocSize;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxMemAllocSize, nullptr));
			double maxMemAllocMB = maxMemAllocSize / (1024.0 * 1024.0);
			std::cout << "        Max memory allocation: " << maxMemAllocMB << " MB" << std::endl;

			// - Размер локальной памяти (быстрая память, разделяемая внутри рабочей группы)
			cl_ulong localMemSize;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, nullptr));
			double localMemKB = localMemSize / 1024.0;
			std::cout << "        Local memory size: " << localMemKB << " KB" << std::endl;

			// - Версия OpenCL, поддерживаемая устройством
			size_t deviceVersionSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &deviceVersionSize));
			std::vector<unsigned char> deviceVersion(deviceVersionSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, deviceVersionSize, deviceVersion.data(), nullptr));
			std::cout << "        OpenCL version: " << deviceVersion.data() << std::endl;
		}
	}

	return 0;
}
