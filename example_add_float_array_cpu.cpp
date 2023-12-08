#include "warp/native/builtin.h"


#ifdef _WIN32
#include <Windows.h>
#define dlsym GetProcAddress
#else
#include <dlfcn.h>
#include <wordexp.h>
#endif


using namespace wp;

// CPU entry points
void (*add_float_arrays_cpu_forward)(launch_bounds_t dim,
array_t<float32> var_dest,
array_t<float32> var_a,
array_t<float32> var_b);


#include <cassert>
#include <vector>
#include <iostream>

// warp-clang.dll entry function pointer signatures
using lookup_func = uint64_t (*)(const char* dll_name, const char* function_name);
using load_obj_func = int (*)(const char* object_file, const char* module_name);

static void expand_environment_strings(const char* src, char* dst, size_t size)
{
    #if defined(_WIN32)
        int ret = ExpandEnvironmentStringsA(src, dst, size);
        assert(ret != 0 && ret <= size);
    #else
        wordexp_t exp_result;
        int ret = wordexp(src, &exp_result, 0);
        assert(ret == 0);
        strncpy(dst, exp_result.we_wordv[0], size);
        dst[size - 1] = '\0';
        wordfree(&exp_result);
    #endif
}

int main(int argc, char* argv[])
{
    #ifdef _WIN32
        const char* cpu_kernel_filename = "%LOCALAPPDATA%/NVIDIA Corporation/warp/Cache/0.9.0/bin/wp___main__.o";
    #else
        const char* cpu_kernel_filename = "~/.cache/warp/0.9.0/bin/wp___main__.o";
    #endif

    if (argc > 1)
    {
        cpu_kernel_filename = argv[1];
    }

    char cpu_kernel_filename_expand[4096];
    expand_environment_strings(cpu_kernel_filename, cpu_kernel_filename_expand, 4096);
    cpu_kernel_filename = cpu_kernel_filename_expand;

    std::cout << "filename:" << cpu_kernel_filename << std::endl;


    #ifdef _WIN32
        const char* warp_clang_dll = "../warp/warp/bin/warp-clang.dll";
    #else
        const char* warp_clang_dll = "../warp/warp/bin/warp-clang.so";
    #endif

    if (argc > 2)
    {
        warp_clang_dll = argv[2];
    }


    char warp_clang_dll_expand[4096];
    expand_environment_strings(warp_clang_dll, warp_clang_dll_expand, 4096);
    warp_clang_dll = warp_clang_dll_expand;

    #ifdef _WIN32
        HMODULE warp_lib = (HMODULE)LoadLibraryA(warp_clang_dll);
    #else
        void* warp_lib = dlopen(warp_clang_dll, RTLD_NOW);
    #endif

    if (!warp_lib)
    {
        std::cout << "Unable to load library " << warp_clang_dll << std::endl << std::endl;
        return false;
    }

    auto *load_obj = reinterpret_cast<load_obj_func>(dlsym(warp_lib, "load_obj"));
    auto *lookup = reinterpret_cast<lookup_func>(dlsym(warp_lib, "lookup"));

    load_obj(cpu_kernel_filename, "kernel_module");

    std::string func_name = "add_float_arrays_cpu_forward";
    add_float_arrays_cpu_forward = reinterpret_cast<decltype(add_float_arrays_cpu_forward)>(lookup("kernel_module", func_name.c_str()));
    if (!add_float_arrays_cpu_forward)
    {
        std::cout << "Unable to get function " << func_name << " from library " << cpu_kernel_filename << std::endl << std::endl;
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
