from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['numpy', 'scipy']

with open("README.md", "r") as h:
    long_description = h.read()

setup(
    name="bayesmtl",
    version="0.0.2",
    author="LLNL",
    description='Bayesian Multitask Learning framework',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        'Topic : Multitask Learning'
    ],
    python_requires='>=3.6',
)