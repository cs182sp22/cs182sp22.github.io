import torch
import traceback

# A quick demo showing what torch.Tensor is

"""
Tensor construction.
Tensors have data, a shape, and a dtype:
"""
print("\033[32m\n==== torch.Tensor Construction ====\033[0m")
my_vector = torch.tensor([1, 2, 3], dtype=torch.float32)
print("my_vector: ", my_vector)
print("my_vector.shape: ", my_vector.shape)

# a matrix with shape=[2, 3] (2 rows, 3 columns)
my_matrix = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
    ],
    dtype=torch.float32
)
print("my_matrix: ", my_matrix)
print("my_matrix.shape: ", my_matrix.shape)

"""
Tensor indexing, slicing
"""
print("\033[32m\n==== indexing, slicing ====\033[0m")
my_matrix = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
    ],
    dtype=torch.float32
)
print("my_matrix: ", my_matrix)
print("my_matrix.shape: ", my_matrix.shape)

# get element at row=0, column=2
print("row=0, col=2: ", my_matrix[0, 2])

# get first row
print("first row: ", my_matrix[0, :])

# assignment
my_matrix[0, 2] = 42
my_matrix[:, 0] = -1
print("my_matrix (post assignment): ", my_matrix)

print("\033[32m\n==== torch.Tensor operations ====\033[0m")
# simple arithmetic operators (+, -, *, /)
matrix_of_ones = torch.tensor(
    [
        [1, 1, 1],
        [1, 1, 1]
    ],
    dtype=torch.float32,
)
# all the same! convenience fns
matrix_of_ones = torch.ones(size=my_matrix.shape, dtype=my_matrix.dtype)
matrix_of_ones = torch.ones_like(my_matrix)

print("before add: ", my_matrix)
print("after add (matrix_of_ones): ", my_matrix + matrix_of_ones)

"""
By default, tensors are created on the CPU device (eg live in your CPU memory)
"""
print("\033[32m\n==== torch.Tensor devices ====\033[0m")
print("my_matrix.device: ", my_matrix.device)

my_matrix_cpu = torch.ones(size=[2, 3], device=torch.device("cpu"))
print("my_matrix_cpu: ", my_matrix_cpu)

# IF you have a GPU device (and pytorch is installed to support cuda), then here I create
# a tensor directly on my GPU device
print("torch.cuda.is_available(): ", torch.cuda.is_available())
my_matrix_gpu = torch.ones(size=[2, 3], device=torch.device("cuda:0"))
print("my_matrix_gpu: ", my_matrix_gpu)

# Tensors must live on the same device to do operations on them:
try:
    print("Trying to add a CPU Tensor to a GPU Tensor")
    foo = my_matrix_cpu + my_matrix_gpu
except Exception as exc:
    print(traceback.format_exc())

# I can also move tensors from one device to another
my_matrix_gpu_to_cpu = my_matrix_gpu.to(device=torch.device("cpu"))
print("my_matrix_gpu_to_cpu: ", my_matrix_gpu_to_cpu)

print("\033[32m\n==== dot products, matrix multiply ====\033[0m")
# Dot products (vec to vec)
vec1 = torch.tensor([2, 3])
vec2 = torch.tensor([2, 1])
print("torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1])): ", vec1, vec2)

# matrix multiply
A = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
    ]
)
B = torch.tensor(
    [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
)
# Recall: A.shape=[2, 3], B.shape=[3, 2], A*B will have shape [2, 2]
print(f"A: {A}\nB: {B}\nAB: {torch.mm(A, B)}")
