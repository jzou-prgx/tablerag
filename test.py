from torch.utils.cpp_extension import load
from torch.utils.cpp_extension import load
load(name="test_cuda", sources=["test_kernel.cu"], verbose=True)
