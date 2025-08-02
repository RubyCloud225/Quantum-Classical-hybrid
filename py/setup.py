from setuptools import setup, Extension, find_packages
import pybind11
import sys
# Ensure that the script is run with Python 3
if sys.version_info.major < 3:
    sys.stderr.write("Error: Python 3 is required to run this setup script.\n")
    sys.exit(1)

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

requirements = parse_requirements('requirements.txt')

ext_modules = [
    Extension(
        'preprocessing',
        sources=[
            'src/DataProcessing/PreprocessingBindings.cpp',
            # Add any other C++ source files your extension needs
        ],
        include_dirs=[
            pybind11.get_include(),
            'src',  # your include path(s)
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],
    )
]

setup(
    name='preprocessing',
    version='0.1',
    author='Your Name',
    author_email='your.email@example.com',
    description='Python bindings for your C++ preprocessing pipeline',
    ext_modules=ext_modules,
    packages=find_packages(),
    install_requires=requirements,
    zip_safe=False,
)