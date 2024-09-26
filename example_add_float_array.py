import numpy as np
import warp as wp

wp.init()

use_cpu = False
if use_cpu:
	device_str = "cpu"
else:
	device_str = "cuda"

mangled_filename = "mangled_names_"+device_str+".txt"

@wp.kernel
def add_float_arrays(dest: wp.array(dtype=wp.float32),
             a: wp.array(dtype=wp.float32),
             b: wp.array(dtype=wp.float32)):

    tid = wp.tid()
    dest[tid] = a[tid]+b[tid]



def example_add_float_arrays(device, n):
   
    dest = wp.zeros(n=n, dtype=wp.float32, device=device)
   
    a = wp.array(np.linspace(0.5, 0.9, n), dtype=wp.float32, device=device)
    b = wp.array(np.linspace(100, 110, n), dtype=wp.float32, device=device)
    print("dir(a)=", dir(a))
    print("a.ndim=",a.ndim)
    print("a.shape=",a.shape)
    print("a.strides=",a.strides)
   
    wp.launch(add_float_arrays, dim=n, inputs=[dest, a, b], device=device)
    print("dest.numpy()=",dest.numpy())
   
example_add_float_arrays(device=device_str, n=8)

device = wp.get_device()
module = wp.get_module("__main__")

# loading the module triggers the hashing process
module.load(device)
print(dir(module))
print(module.__dir__())
# get module identifier
module_hash_suffix = f"{module.hash_module().hex()[:7]}"
module_identifier = f"wp_{module.name}_{module_hash_suffix}"


if use_cpu:
	output_name = "module_codegen.o"
	kernel_forward = f"{add_float_arrays.get_mangled_name()}_cpu_forward"
	kernel_backward = f"{add_float_arrays.get_mangled_name()}_cpu_backward"
else:
	output_arch = min(device.arch, wp.config.ptx_target_arch)
	output_name = f"module_codegen.sm{output_arch}.ptx"
	# get kernel entry points
	kernel_forward = f"{add_float_arrays.get_mangled_name()}_cuda_kernel_forward"
	kernel_backward = f"{add_float_arrays.get_mangled_name()}_cuda_kernel_backward"


mangled_names = []

mangled_names.append(wp.config.kernel_cache_dir)
mangled_names.append(module_identifier)
mangled_names.append(output_name)            
mangled_names.append(kernel_forward)
mangled_names.append(kernel_backward)

with open(mangled_filename, "w") as file:
	for string in mangled_names:
				print(string)
				file.write(string + "\n")
        