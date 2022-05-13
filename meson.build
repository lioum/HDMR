project('hdmr', 'cpp', 'cuda', default_options : ['cpp_std=c++20'])

cuda = import('unstable-cuda')

cuda_comp = meson.get_compiler('cuda')
nvcc_flags = cuda.nvcc_arch_flags(cuda_comp, 'Pascal')

readable = cuda.nvcc_arch_readable(cuda_comp, 'Pascal')
message('Building for architectures ' + ' '.join(readable))

includes = [include_directories('include')]

subdir('src')

eigen = dependency('eigen3')

stb = static_library('stb', sources: sources_stb, include_directories: includes)

executable('hdmr_cpu', dependencies: eigen, sources: sources_cpu, link_with: stb, include_directories: includes)
executable('hdmr_gpu', cuda_args: nvcc_flags, dependencies: eigen, sources: sources_gpu, link_with: stb, include_directories: includes)
