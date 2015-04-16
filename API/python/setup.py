from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os.path as path

# module1 = Extension("overfeat",
#                     include_dirs = ['../../src', numpy.get_include()],
#                     library_dirs = ['/usr/local/src/torch/install/lib','../../src','/opt/OpenBLAS/lib'],
#                     libraries = ['TH', 'overfeat', 'openblas'],
#                     sources = ['overfeatmodule.cpp'],
#                     extra_compile_args=['-fopenmp'],
#                     #extra_link_args=['-lgomp', '-lTH', '-L%s'%(path.abspath('../../src/libTH.a'))])
# 		    extra_link_args=['-lgomp'])

modules = cythonize(
    [Extension("overfeat", ["overfeat.pyx", "of_util.cpp"], include_dirs = ['../../src', numpy.get_include()],
               library_dirs = ['/usr/local/src/torch/install/lib','../../src','/opt/OpenBLAS/lib'],
               libraries = ['TH', 'overfeat', 'openblas'],
               language="c++",
               extra_compile_args=['-fopenmp','-std=c++11'],extra_link_args=['-lgomp'])])

setup(name = 'overfeat',
      version = '1.0',
      description = 'Python bindings for overfeat',
      ext_modules = modules,
      install_requires = ['numpy'])

