from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(packages=find_packages(where='./src'), long_description=open('README.md').read(),
        long_description_content_type='text/markdown')
