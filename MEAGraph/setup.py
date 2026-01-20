import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="MEAG_VAE",
    version="0.0.1",
    author="Hong Sun",
    author_email="sun36@llnl.gov",
    description="MEAGraph: Multi-kernel Edge Attention Graph Autoencoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'yacs',
        'torch',
        'torch-geometric',
        'numpy',
        'tensorboardx',
        'ase',
        'tqdm',
        'dscribe',
        'networkx',
        'matplotlib',
        'pandas',
        'scikit-learn',
        'pyyaml'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)