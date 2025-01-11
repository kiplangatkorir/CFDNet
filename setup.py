from setuptools import setup, find_packages

setup(
    name='cfdnet',
    version='0.1.0',
    description='A Continuous Function Decomposition Network for sequence modeling',
    author='Your Name',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'scipy'
    ],
    python_requires='>=3.8',
)
