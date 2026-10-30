import os
from glob import glob
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__author__ = 'Florian Hahlbohm'
__description__ = 'CUDA backend of ZipNeRF: runtime-compiled multisampled hash grid featurization with fused MLPs, sampling, and losses.'

ENABLE_FASTMATH = True  # set to False to disable fast math optimizations (e.g., for debugging)
ENABLE_NVCC_LINEINFO = False  # set to True for profiling kernels with Nsight Compute (overhead is minimal)

module_root = Path(__file__).parent.absolute()
extension_name = module_root.name
extension_root = module_root / extension_name
cuda_modules = [d.name for d in Path(extension_root).iterdir() if d.is_dir() and d.name != 'torch_bindings']

# gather source files and include directories (the featurization kernels in featurization/rtc are compiled at runtime)
sources = [str(extension_root / 'torch_bindings' / 'bindings.cu')]
include_dirs = []
for module in cuda_modules:
    sources += glob(str(extension_root / module / 'src' / '**' / '*.cu'), recursive=True)
    include_dirs.append(str(extension_root / module / 'include'))

# set up compiler flags
cxx_flags = ['/std:c++17' if os.name == 'nt' else '-std=c++17']
nvcc_flags = ['-std=c++17', '--expt-relaxed-constexpr']
if ENABLE_FASTMATH:
    cxx_flags.append('-O3')
    nvcc_flags.append('-O3')
    nvcc_flags.append('-use_fast_math')
if ENABLE_NVCC_LINEINFO:
    nvcc_flags.append('-lineinfo')

# define the CUDA extension
extension = CUDAExtension(
    name=f'{extension_name}._C',
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': cxx_flags,
        'nvcc': nvcc_flags
    },
    libraries=['cuda', 'nvrtc', 'cublasLt'],
)

# set up the package
setup(
    name=extension_name,
    author=__author__,
    packages=[f'{extension_name}.torch_bindings'],
    ext_modules=[extension],
    description=__description__,
    cmdclass={'build_ext': BuildExtension}
)
