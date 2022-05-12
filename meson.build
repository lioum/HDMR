project('hdmr', 'cpp', 'cuda', default_options : ['cpp_std=c++20'])

includes = [include_directories('include')]

subdir('src')

eigen = dependency('eigen3')

executable('hdmr_cpu', dependencies: eigen, sources: sources_cpu, include_directories: includes)
executable('hdmr_gpu', dependencies: eigen, sources: sources_gpu, include_directories: includes)
