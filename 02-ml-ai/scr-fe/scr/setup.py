from setuptools import setup

setup(
    name='scr',
    version='0.1',
    py_modules=['scr'],
    include_package_data=True,
    install_requires=[
        'click',
    ],
    entry_points='''
        [console_scripts]
        scr=main:cli
    ''',
)
