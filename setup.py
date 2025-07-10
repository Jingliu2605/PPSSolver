import os
from os.path import isfile
from setuptools import setup, Extension, find_packages


# Define Cython folders and extensions
cython_folders = [
    "problem", "operators", "solvers", "pymoo/model", "pymoo/nds", "pymoo/termination",
    "pymoo/util", "pymoo/algorithms", "pymoo/operators", "pymoo/operators/crossover",
    "pymoo/operators/mutation", "pymoo/operators/sampling", "pymoo/operators/repair",
    "pymoo/operators/selection", "pymoo/performance_indicator"
]

pyx_files = []
extensions = []
for folder in cython_folders:
    compile_files = [f for f in os.listdir(folder) if isfile(os.path.join(folder, f)) and f.endswith(".pyx")]
    for file in compile_files:
        module = Extension(
            f"{folder.replace('/', '.')}.{os.path.splitext(file)[0]}",
            [os.path.join(folder, file)],
            include_dirs=[
                'C:/PPSSP/SoftwareX/portfolio_optimization-master/venv/Lib/site-packages/numpy/core/include'
            ]
        )
        pyx_file = f"{folder.replace('/', '.')}.{os.path.splitext(file)[0]}"
        pyx_files.append(pyx_file)
        extensions.append(module)


# Define the setup configuration
setup(
    name="portfolio_optimization",
    version="1.0.0",
    description="An open-source software tool for Project Portfolio Selection and Scheduling Problem",
    author="Jing Liu",
    author_email="liujing2605@gmail.com",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas"
    ],
    ext_modules=extensions,
    entry_points={
        "console_scripts": [
            "portfolio_gui=GUI:main",  # Replace `main` with the entry point function in GUI.py
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

# import os
# from os.path import isfile
# from setuptools import setup, Extension
#
#
# cython_folders = ["problem", "operators", "solvers", "pymoo/model", "pymoo/nds", "pymoo/termination",
#                   "pymoo/util", "pymoo/algorithms", "pymoo/operators", "pymoo/operators/crossover",
#                   "pymoo/operators/mutation", "pymoo/operators/sampling", "pymoo/operators/repair",
#                   "pymoo/operators/selection", "pymoo/performance_indicator"]
#
# for folder in cython_folders:
#     compile_files = [f for f in os.listdir(folder) if isfile(os.path.join(folder, f)) and
#                      (f.endswith(".pyx"))]
#     for file in compile_files:
#         module = Extension(f"{folder.replace('/', '.')}.{os.path.splitext(file)[0]}",
#                            [os.path.join(folder, file)], include_dirs=[
#                                'C:/PPSSP/SoftwareX/portfolio_optimization-master/venv/Lib/site-packages/numpy/core/include'])
#         setup(
#             name='cythonTest',
#             version='1.0',
#             author='jetbrains',
#             ext_modules=[module]
#         )
# #