import numpy as np
import warp as wp

wp.init()
device = "cpu"

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
   
example_add_float_arrays(device=device, n=8)
