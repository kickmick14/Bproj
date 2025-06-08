import io
from setuptools import setup, find_packages

# Read the long description from your README
with io.open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bproj",
    version="0.1.0",
    author="Michael Kane",
    author_email="michaelkane14@icloud.com",
    description="A project for the spare time",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kickmick14/Bproj",
    packages=find_packages(exclude=["tests", "docs"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scitkit-learn",
        "requests",
        "python-binance>=1.0.16",
        "tensorflow",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)