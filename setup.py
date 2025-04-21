from setuptools import setup, find_packages

setup(
    name="jaxamr",
    version="0.1",
    author="Haocheng Wen",
    packages=find_packages(where="src"),
    package_dir={"": "src"}
)
