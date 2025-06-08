from setuptools import setup, find_packages

setup(
    name="bproj",
    version="0.1.0",
    package_dir={"": "bproj"},           # point packages to the bproj/ folder
    packages=find_packages(where="bproj")
)