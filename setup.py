from setuptools import setup, find_packages

setup(
    name='liriscat',
    version='1.0.0',
    author='Arthur BATEL',
    author_email='arthur.batel@insa-lyon.fr',
    packages=find_packages(),
    description="""Liriscat""",
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    url='https://github.com/arthur-batel/liriscat.git',
    install_requires=[
    ],  # And any other dependencies foo needs
    entry_points={
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.11",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires='>=3.6',
)
