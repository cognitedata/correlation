import re

from setuptools import find_packages, setup

version = re.search('^__version__\s*=\s*"(.*)"', open("cognite/correlation/__init__.py").read(), re.M).group(1)

setup(
    name="cognite-correlation",
    version=version,
    packages=["cognite." + p for p in find_packages(where="cognite")],
    install_requires=["pandas", "numpy", "matplotlib"],
    python_requires=">=3.5",
)
