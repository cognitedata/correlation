import re
from os import path

from setuptools import find_packages, setup

version = re.search('^__version__\s*=\s*"(.*)"', open("cognite/correlation/__init__.py").read(), re.M).group(1)

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cognite-correlation",
    version=version,
    packages=["cognite." + p for p in find_packages(where="cognite")],
    install_requires=["pandas", "numpy", "matplotlib"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.5",
)
