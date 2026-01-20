from setuptools import setup, find_packages

# long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

#
setup(
    name='iniabu',
    version='0.3.1',
    url='https://github.com/LLNL/iniabu',
    license='GPL-2.0',
    author='Reto Trappitsch',
    author_email='trappitsch1@llnl.gov',
    description='Solar System initial isotopic abundance reader.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # add the data files into the system folder
    data_files=[('iniabu_data', ['iniabu/iniabu_data/lodders09.dat'])],
    packages=find_packages(
        include=["iniabu", "iniabu.*"]
    ),  # include all packages under iniabu
    package_data={
        "": ["*.dat"],  # include data
    },
    install_requires=['numpy'],
    # classifiers
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ]
)
