# warp_cpp
Examples calling [NVIDIA Warp](https://github.com/nvidia/warp) precompiled (cached) kernels directly from C++ (without Python)

## Usage

Install Warp

```
pip install numpy
git clone https://github.com/NVIDIA/warp.git
cd warp
python build_lib.py --cuda_path=/usr/local/warp
pip install -e .
```
Run the Warp Python example to jit compile the example_add_float_array.py
```
python example_add_float_array.py
Warp 0.8.2 initialized:
   CUDA Toolkit: 11.8, Driver: 12.1
   Devices:
     "cpu"    | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
     "cuda:0" | NVIDIA GeForce RTX 4090 Laptop GPU (sm_89)
   Kernel cache: C:\Users\erwin\AppData\Local\NVIDIA Corporation\warp\Cache\0.8.2
Module __main__ load on device 'cpu' took 15.43 ms
dest.numpy()= [100.5      101.98572  103.47143  104.95714  106.442856 107.92857
 109.41428  110.9     ]
 ```
Note the Kernel cache path, it will contain the compiled Warp kernel as CPU DLL or CUDA PTX binary.

Use cmake, compile and run the C++ example_add_float_array.cpp
```
cmake .
cmake --build .
Debug\example_add_float_array.exe
a:1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8
b:100 200 300 400 500 600 700 800
Sum:101.1 202.2 303.3 404.4 505.5 606.6 707.7 808.8
```
