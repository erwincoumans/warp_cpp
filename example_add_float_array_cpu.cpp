//todo: only tested/developed on Windows, test/fix Linux version
//todo: remove hardcoded path, let user/dev specify path through build system (CMake)
#include "warp/native/builtin.h"


#ifdef _WIN32
#include <Windows.h>
#define dlsym GetProcAddress
#else
#include <dlfcn.h>
#endif

//todo: remove hardcoded path, let user/dev specify path through command-line arguments
#define WARP_CPU_TEST_KERNEL "C:/Users/erwin/AppData/Local/NVIDIA Corporation/warp/Cache/0.8.2/bin/wp___main__.dll"

using namespace wp;

// CPU entry points
void (*add_float_arrays_cpu_forward)(launch_bounds_t dim,
array_t<float32> var_dest,
array_t<float32> var_a,
array_t<float32> var_b);


#include <vector>
#include <iostream>

int main(int argc, char* argv[])
{
#ifdef _WIN32
    //module depends on warp.dll, so point to its location
    SetDllDirectory("D:/dev/warp_cpp/third_party/warp/warp/bin");
    //load the DLL module that contains the kernel
    HMODULE warp_lib = (HMODULE)LoadLibraryA(WARP_CPU_TEST_KERNEL);
#else
    //todo linux
    void* warp_lib = dlopen(WARP_CPU_TEST_KERNEL, RTLD_NOW);
#endif
    if (!warp_lib) {
        std::cout << "Unable to load library " << WARP_CPU_TEST_KERNEL << std::endl << std::endl;
        return false;
    }
    std::string func_name = "add_float_arrays_cpu_forward";
    add_float_arrays_cpu_forward = reinterpret_cast<decltype(add_float_arrays_cpu_forward)> (dlsym(warp_lib, func_name.c_str()));
    if (!add_float_arrays_cpu_forward)
    {
        std::cout << "Unable to get function " << func_name << " from library " << WARP_CPU_TEST_KERNEL << std::endl << std::endl;
        return false;
    }
    wp::launch_bounds_t bounds;
    bounds.ndim = 1;
    bounds.shape[0] = 8;//?
    bounds.size = 8;

    wp::array_t<float> var_dest, var_a, var_b;

    std::vector<float> my_data_dest = { -1.,-1.,-1.,-1.,-1.,-1.,-1.,-1. };
    std::vector<float> my_data_a = { 1.1,2.2,3.3,4.4,5.5,6.6,7.7, 8.8 };
    std::vector<float> my_data_b = { 100.,200.,300.,400.,500.,600.,700.,800. };

    int size = my_data_dest.size();

    var_dest.ndim = 1;
    var_dest.shape[0] = size;
    var_dest.strides[0] = sizeof(float);
    var_dest.data = &my_data_dest[0];

    var_a.ndim = 1;
    var_a.shape[0] = size;
    var_a.strides[0] = sizeof(float);
    var_a.data = &my_data_a[0];

    var_b.ndim = 1;
    var_b.shape[0] = size;
    var_b.strides[0] = sizeof(float);
    var_b.data = &my_data_b[0];

    std::cout << "a:";
    for (auto const& c : my_data_a)
        std::cout << c << ' ';
    std::cout << std::endl;
    std::cout << "b:";
    for (auto const& c : my_data_b)
        std::cout << c << ' ';
    std::cout << std::endl;

    add_float_arrays_cpu_forward(bounds, var_dest, var_a, var_b);

    std::cout << "Sum:";

    for (auto const& c : my_data_dest)
        std::cout << c << ' ';
    std::cout << std::endl;    
}
