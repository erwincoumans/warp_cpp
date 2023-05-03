#include <iostream>
#include <fstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#define dlsym GetProcAddress
#define DYNAMIC_CUDA_PATH "nvcuda.dll"
#else
#include <dlfcn.h>
#define DYNAMIC_CUDA_PATH "/usr/lib/x86_64-linux-gnu/libcuda.so"
#endif

using namespace std;

enum TINY_CUDA_CODES
{
    CUDA_SUCCESS = 0,
    CU_GET_PROC_ADDRESS_DEFAULT = 0,
    cudaEnableDefault = 0,
};

// CUDA driver API functions.
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct cudaArray_st* cudaArray_t;
typedef struct CUctx_st* CUcontext;

typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef int CUdevice_v1;
typedef CUdevice_v1 CUdevice;

#if defined(_WIN64) || defined(__LP64__)
typedef unsigned long long CUdeviceptr_v2;
#else
typedef unsigned int CUdeviceptr_v2;
#endif
typedef CUdeviceptr_v2 CUdeviceptr;                          /**< CUDA device pointer */


// see https://docs.nvidia.com/cuda/cuda-runtime-api/driver-vs-runtime-api.html#driver-vs-runtime-api
// cuda driver (cuda.so)
TINY_CUDA_CODES(*cuDriverGetVersion)(int* version);
TINY_CUDA_CODES(*cuInit)(unsigned int flags);
TINY_CUDA_CODES(*cuDeviceGetCount)(int* count);
TINY_CUDA_CODES(*cuDeviceGetCount2)(int* count);
TINY_CUDA_CODES(*cuGetProcAddress) (const char* symbol, void** pfn, int  cudaVersion, uint64_t flags);
TINY_CUDA_CODES(*cuMemcpyHtoD)(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
TINY_CUDA_CODES(*cuMemcpyDtoH)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);

//TINY_CUDA_CODES(*cuMemcpyFromArray)  (void* dst, const cudaArray_t src, size_t wOffset, size_t hOffset, size_t count, unsigned int kind);



TINY_CUDA_CODES(*cuModuleLoadData)(CUmodule* module, const void* image);
TINY_CUDA_CODES (*cuModuleGetFunction)(CUfunction* hfunc, CUmodule 	hmod, const char* name);
TINY_CUDA_CODES (*cuDeviceGet)(CUdevice* device, int 	ordinal);
TINY_CUDA_CODES (*cuCtxCreate)(CUcontext* pctx, unsigned int flags, CUdevice dev);
TINY_CUDA_CODES (*cuLaunchKernel)(CUfunction 	f, unsigned int 	gridDimX,unsigned int 	gridDimY,unsigned int 	gridDimZ,
    unsigned int 	blockDimX,    unsigned int 	blockDimY,    unsigned int 	blockDimZ,    unsigned int 	sharedMemBytes,
    cudaStream_t 	hStream,    void** kernelParams,    void** extra);
TINY_CUDA_CODES (*cuMemAlloc)(CUdeviceptr* dptr, size_t 	bytesize);
TINY_CUDA_CODES(*cuMemFree)(CUdeviceptr dptr);

enum  cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,      /**< Host   -> Host */
    cudaMemcpyHostToDevice = 1,      /**< Host   -> Device */
    cudaMemcpyDeviceToHost = 2,      /**< Device -> Host */
    cudaMemcpyDeviceToDevice = 3,      /**< Device -> Device */
    cudaMemcpyDefault = 4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};


#define LOAD_CUDA_FUNCTION(name, version) \
  name = reinterpret_cast<decltype(name)>(dlsym(cuda_lib , #name version)); \
  if (!name) cout << "Error:" << #name <<  " not found in CUDA library" << endl


// Utility function to read the contents of a file into a string
std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}


const int LAUNCH_MAX_DIMS = 4;   // should match types.py

struct launch_bounds_t
{
    int shape[LAUNCH_MAX_DIMS]; // size of each dimension
    int ndim;                   // number of valid dimension
    size_t size;                // total number of threads
};

template <typename T>
struct CudaVector
{
    CudaVector()
        :m_cudaMem(0), m_ownsCudaPtr(false)
    {

    }

    virtual ~CudaVector()
    {
        releaseCuda();
    }

    void releaseCuda()
    {
        if (m_cudaMem && m_ownsCudaPtr)
        {
            cuMemFree((CUdeviceptr)m_cudaMem);
            m_cudaMem = 0;
            m_ownsCudaPtr = false;
        }
    }

    void setCudaPtr(T* cuda_ptr)
    {
        releaseCuda();
        m_cudaMem = cuda_ptr;
        m_ownsCudaPtr = false;
    }

    void resize(int size)
    {
        if (m_hostMem.size() != size || !m_ownsCudaPtr)
        {
            releaseCuda();
            auto res = cuMemAlloc((CUdeviceptr*)&m_cudaMem, sizeof(T) * size);
            m_ownsCudaPtr = true;
            m_hostMem.resize(size);
        }
    }
    void copyToCuda()
    {
        T* dst = m_cudaMem;
        T* src = &(m_hostMem[0]);
        cuMemcpyHtoD((CUdeviceptr)dst, src, sizeof(T) * m_hostMem.size());
    }
    void copyToCpu()
    {
        T* dst = &(m_hostMem[0]);
        T* src = m_cudaMem;
        cuMemcpyDtoH(dst, (CUdeviceptr)src,sizeof(T) * m_hostMem.size());
    }
    size_t sizeInBytes() const
    {
        return sizeof(T) * m_hostMem.size();
    }
    T* m_cudaMem;
    bool m_ownsCudaPtr;
    std::vector<T> m_hostMem;
};


int main(int argc, char* argv[])
{
#ifdef _WIN32
    HMODULE cuda_lib = (HMODULE)LoadLibraryA(DYNAMIC_CUDA_PATH);
#else
    void* cuda_lib = dlopen(DYNAMIC_CUDA_PATH, RTLD_NOW);
#endif
    if (!cuda_lib) {
        cout << "Unable to load library " << DYNAMIC_CUDA_PATH << endl << endl;
        return false;
    }

    cout << "hello cuda world" << endl;

    LOAD_CUDA_FUNCTION(cuDriverGetVersion, "");
    LOAD_CUDA_FUNCTION(cuInit, "");
    LOAD_CUDA_FUNCTION(cuDeviceGetCount, "");
    LOAD_CUDA_FUNCTION(cuGetProcAddress, "");
    LOAD_CUDA_FUNCTION(cuMemcpyHtoD, "");
    LOAD_CUDA_FUNCTION(cuMemcpyDtoH, "");
    
    LOAD_CUDA_FUNCTION(cuModuleLoadData, "");
    LOAD_CUDA_FUNCTION(cuModuleGetFunction, "");
    LOAD_CUDA_FUNCTION(cuDeviceGet, "");
    LOAD_CUDA_FUNCTION(cuCtxCreate , "");
    LOAD_CUDA_FUNCTION(cuLaunchKernel, "");
    LOAD_CUDA_FUNCTION(cuMemAlloc, "");
    LOAD_CUDA_FUNCTION(cuMemFree, "");


    auto result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to initialize CUDA driver API" << std::endl;
        return 1;
    }

    int cuda_driver_version;
    result = cuDriverGetVersion(&cuda_driver_version);
    cout << "CUDA driver version:" << cuda_driver_version << endl;
    int device_count = 0;
    result = cuDeviceGetCount(&device_count);
    cout << "CUDA device count:" << device_count << endl;
    result = cuGetProcAddress("cuDeviceGetCount", (void**)&cuDeviceGetCount2, cuda_driver_version, CU_GET_PROC_ADDRESS_DEFAULT);
    if (CUDA_SUCCESS != result)
    {
        cout << "cuDeviceGetCount not found" << endl;
        exit(1);
    }

   

    // Get the default CUDA device
    CUdevice cuDevice;
    result = cuDeviceGet(&cuDevice, 0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to get CUDA device" << std::endl;
        return 1;
    }

    // Create a CUDA context
    CUcontext cuContext;
    result = cuCtxCreate(&cuContext, 0, cuDevice);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to create CUDA context" << std::endl;
        return 1;
    }


    // Load the PTX file
    std::string ptxSource = readFile("C:/Users/erwin/AppData/Local/NVIDIA Corporation/warp/Cache/0.8.2/bin/wp___main__.sm70.ptx");
    printf("len=%d\n", ptxSource.length());

    // Create a CUDA module from the PTX source
    CUmodule cuModule;
    result = cuModuleLoadData(&cuModule, ptxSource.c_str());
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to load CUDA module" << std::endl;
        return 1;
    }

    // Get the kernel function from the module
    CUfunction cuFunction;
    result = cuModuleGetFunction(&cuFunction, cuModule, "add_float_arrays_cuda_kernel_forward");
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to get CUDA function" << std::endl;
        return 1;
    }

    int num_items = 8;

    CudaVector< launch_bounds_t> bounds_cuda;
    bounds_cuda.resize(1);
    bounds_cuda.m_hostMem[0].ndim = 1;
    bounds_cuda.m_hostMem[0].shape[0] = num_items;//??
    bounds_cuda.m_hostMem[0].size = num_items;
    bounds_cuda.copyToCuda();

    CudaVector<float> a, b, c;
    a.resize(num_items);
    a.m_hostMem = { 1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8 };
    b.resize(num_items);
    b.m_hostMem = { 100.,200.,300.,400.,500.,600.,700.,800. };
    c.resize(num_items);
    c.m_hostMem = { -1.,-1., -1., -1., -1., -1., -1., -1. };

    a.copyToCuda();
    b.copyToCuda();
    c.copyToCuda();

    // Set up kernel arguments
    void* args[] = { &bounds_cuda.m_cudaMem, &c.m_cudaMem , &a.m_cudaMem, &b.m_cudaMem };

    int numThreadsPerBlock = 64;
    int numPairsPerBlock = numThreadsPerBlock / 4;
    int numBlocks = (num_items + (numPairsPerBlock - 1)) / numPairsPerBlock;
    //numBlocks = 1;
    //numThreadsPerBlock = 1;
    // Launch the kernel
    result = cuLaunchKernel(cuFunction, 
        numBlocks, 1, 1, //grid
        numThreadsPerBlock, 1, 1, //block
        0, nullptr, args, nullptr);

    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to launch CUDA kernel" << std::endl;
        return 1;
    }

    a.copyToCpu();
    b.copyToCpu();
    c.copyToCpu();

    std::cout << "Sum:";

    for (auto const& c : c.m_hostMem)
        std::cout << c << ' ';
    std::cout << std::endl;

}