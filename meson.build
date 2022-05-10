project('hdmr', 'cpp', default_options : ['cpp_std=c++20'])

sources = []
includes = [include_directories('include')]

subdir('src')

eigen = dependency('eigen3')

executable('hdmr', dependencies: eigen, sources: sources, include_directories: includes)
