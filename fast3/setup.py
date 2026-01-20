import setuptools
import os

PACKAGE = 'fast2'

here = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(here, os.pardir))


setuptools.setup(
    name=f'{PACKAGE}',
    version="1.0.0",
    author="Junhyug Noh",
    author_email="noh1@llnl.gov",
    description="FASTv2 Fusion Models",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "colorlog",
        "fvcore",
        "numpy",
        "pandas",
        "scipy",
        "h5py",
        "PyYAML",
        "scikit-learn",
        "torch",
        "torch-geometric",
        "torch-scatter",
    ]
)
