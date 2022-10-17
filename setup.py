try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name="seclea_ai",
    version="0.1.2",
    author="Seclea Maintainers",
    author_email="support@seclea.com",
    description="Seclea integration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seclea/seclea_ai",
    packages=find_packages(exclude=["**test", "**example_files"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "circuitbreaker>=1.4.0",
        "decorator>=5.1.1",
        "dill>=0.3.4",
        "pandas>=1.3.0",
        "pympler>=1.0.1",
        "pyyaml>=6.0",
        "requests>=2.0.0",
        "wrapt>=1.14.1",
        "zstandard>=0.15.2",
    ],
)
