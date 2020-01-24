import numpy as np


seq_gpu_n32 = np.array([10, 0.436, 100, 0.609, 1000, 4.183])
seq_gpu_n64 = np.array([10, 0.577, 100, 3.692, 1000, 34.909])

seq_cpu_n32 = np.array([10, 0.011, 100, 0.008, 1000, 0.061])
seq_cpu_n64 = np.array([10, 0.011, 100, 0.055, 1000, 0.510])

naive_cpu_n64 = np.array([10, 0.023, 100, 0.012, 1000, 0.077])
naive_cpu_n128 = np.array([10, 0.022, 100, 0.130, 1000, 1.175])
naive_cpu_n256 = np.array([10, 0.131, 100, 1.070, 1000, 10.468])
naive_cpu_n512 = np.array([10, 1.226, 100, 10.912, 1000, 108.010])

naive_gpu_n64 = np.array([10, 0.328, 100, 0.225, 1000, 0.238])
naive_gpu_n128 = np.array([10, 0.262, 100, 0.269, 1000, 0.337])
naive_gpu_n256 = np.array([10, 0.598, 100, 0.647, 1000, 1.207])
naive_gpu_n512 = np.array([10, 3.207, 100, 3.663, 1000, 8.227])


arr = naive_gpu_n512
length = arr.shape[0] // 2

print("naive_gpu_n512")
for i in range(length):
    print("(%d,%f)" % (arr[2*i], arr[2*i+1]))
